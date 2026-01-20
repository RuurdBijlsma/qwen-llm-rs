#![allow(clippy::missing_errors_doc)]

use clap::Parser;
use color_eyre::eyre::Context;
use color_eyre::eyre::{eyre, Result};
use gag::Gag;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::Special;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use log::info;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::Path;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct MultimodalModel {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl MultimodalModel {
    pub fn load(model_path: &str, n_gpu_layers: i32) -> Result<Self> {
        let _gag_out = Gag::stdout().map_err(|_| eyre!("Unable to gag stdout"))?;
        let _gag_err = Gag::stderr().map_err(|_| eyre!("Unable to gag stderr"))?;

        let backend = LlamaBackend::init().context("Failed to init backend")?;
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers.try_into().unwrap_or(0));

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("Failed to load model")?;

        Ok(Self { backend, model })
    }

    pub fn new_session(&'_ self, mmproj_path: &str, ctx_size: u32) -> Result<MultimodalSession<'_>> {
        MultimodalSession::new(&self.backend, &self.model, mmproj_path, ctx_size)
    }
}

pub struct MultimodalSession<'a> {
    model: &'a LlamaModel,
    context: LlamaContext<'a>,
    mtmd_ctx: MtmdContext,
    n_past: i32,
    ctx_size: usize,
}

impl<'a> MultimodalSession<'a> {
    fn new(
        backend: &'a LlamaBackend,
        model: &'a LlamaModel,
        mmproj_path: &str,
        ctx_size: u32,
    ) -> Result<Self> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size))
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            .with_n_threads(8)
            .with_n_batch(4096)
            .with_n_ubatch(4096);

        let context = model
            .new_context(backend, ctx_params)
            .context("Failed to create context")?;

        let mtmd_params = MtmdContextParams {
            use_gpu: true,
            print_timings: false,
            n_threads: 8,
            media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())?,
        };

        let mtmd_ctx = MtmdContext::init_from_file(mmproj_path, model, &mtmd_params)
            .context("Failed to initialize MTMD context")?;

        Ok(Self {
            model,
            context,
            mtmd_ctx,
            n_past: 0,
            ctx_size: ctx_size as usize,
        })
    }

    pub fn reset(&mut self) {
        self.context.clear_kv_cache();
        self.n_past = 0;
    }

    pub fn chat<P: AsRef<Path>>(&mut self, prompt: &str, image_paths: &[P]) -> Result<String> {
        let mut result = String::new();
        let stream = self.stream_chat(prompt, image_paths)?;

        for token_res in stream {
            match token_res {
                Ok(token) => result.push_str(&token),
                Err(e) => return Err(e.wrap_err(format!("Error during generation: {result}"))),
            }
        }
        Ok(result)
    }

    pub fn stream_chat<P: AsRef<Path>>(
        &mut self,
        prompt: &str,
        image_paths: &[P],
    ) -> Result<ResponseStream<'a, '_>> {
        let mut bitmaps = Vec::new();
        for p in image_paths {
            let path_str = p.as_ref().to_str().ok_or_else(|| eyre!("Invalid path"))?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd_ctx, path_str)?);
        }

        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let mut full_prompt = prompt.to_string();
        if !bitmaps.is_empty() && !full_prompt.contains(&default_marker) {
            full_prompt = format!("{default_marker} {full_prompt}");
        }

        let messages = vec![LlamaChatMessage::new("user".to_string(), full_prompt)?];
        let chat_template = self.model.chat_template(None)?;
        let formatted_prompt = self.model.apply_chat_template(&chat_template, &messages, true)?;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        // FIX: tokenize expects a slice of references (&[&MtmdBitmap])
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;

        self.n_past = chunks.eval_chunks(
            &self.mtmd_ctx,
            &self.context,
            self.n_past,
            0,
            4096,
            true,
        )?;

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(-1, 1.0, 0.0, 1.5),
            LlamaSampler::top_p(0.8, 1),
            LlamaSampler::temp(0.7),
            LlamaSampler::greedy(),
        ]);

        let batch = LlamaBatch::new(self.ctx_size, 1);

        Ok(ResponseStream {
            model: self.model,
            context: &mut self.context,
            sampler,
            batch,
            n_past: &mut self.n_past,
            is_done: false,
        })
    }
}

pub struct ResponseStream<'a, 'b> {
    model: &'a LlamaModel,
    context: &'b mut LlamaContext<'a>,
    sampler: LlamaSampler,
    batch: LlamaBatch<'a>,
    n_past: &'b mut i32,
    is_done: bool,
}

impl Iterator for ResponseStream<'_, '_> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done { return None; }

        let token = self.sampler.sample(self.context, -1);
        self.sampler.accept(token);

        if self.model.is_eog_token(token) {
            self.is_done = true;
            return None;
        }

        let piece = match self.model.token_to_str(token, Special::Tokenize) {
            Ok(s) => s,
            Err(e) => return Some(Err(eyre!(e))),
        };

        self.batch.clear();
        if let Err(e) = self.batch.add(token, *self.n_past, &[0], true) {
            return Some(Err(eyre!(e)));
        }

        *self.n_past += 1;

        if let Err(e) = self.context.decode(&mut self.batch) {
            return Some(Err(eyre!("Decode failed: {}", e)));
        }

        Some(Ok(piece))
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "assets/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf")]
    model: String,
    #[arg(long, default_value = "assets/qwen3vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf")]
    mmproj: String,
    #[arg(long, default_value_t = 99)]
    n_gpu_layers: i32,
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,
}

fn main() -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();
    color_eyre::install()?;

    let args = Args::parse();

    let model_manager = MultimodalModel::load(&args.model, args.n_gpu_layers)?;
    let mut session = model_manager.new_session(&args.mmproj, args.ctx_size)?;

    info!("--- Session Initialized ---");

    let img_hike = Path::new("assets/img/hike.png");
    let img_farm = Path::new("assets/img/farm.png");
    let img_torus = Path::new("assets/img/torus.png");
    let prompt = "Caption this image in one paragraph. Respond with the caption only.";

    info!("Hike: {}", session.chat(prompt, &[img_hike])?);

    session.reset();
    info!("Farm: {}", session.chat(prompt, &[img_farm])?);

    session.reset();
    info!("Torus: {}", session.chat(prompt, &[img_torus])?);

    session.reset();
    info!("Hike 2: {}", session.chat(prompt, &[img_hike])?);
    info!(
        "Follow up: {}",
        session.chat("Where might this be?", &[] as &[&str])?
    );

    Ok(())
}