use clap::Parser;
use color_eyre::eyre::Context;
use color_eyre::eyre::{eyre, Result};
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
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// -------------------------------------------------------------------------
// 1. The Session Struct
// -------------------------------------------------------------------------
pub struct MultimodalSession<'a> {
    model: &'a LlamaModel,
    context: LlamaContext<'a>,
    // FIX 1: MtmdContext does not take a lifetime parameter in your version
    mtmd_ctx: MtmdContext,
    n_past: usize,
    ctx_size: usize,
}

impl<'a> MultimodalSession<'a> {
    pub fn new(
        backend: &'a LlamaBackend,
        model: &'a LlamaModel,
        mmproj_path: &str,
        ctx_size: u32,
    ) -> Result<Self> {
        // Create Context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size))
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            .with_n_threads(8)
            .with_n_batch(2048)
            .with_n_ubatch(512);

        let context = model
            .new_context(backend, ctx_params)
            .context("Failed to create context")?;

        // Init MTMD Context
        let mtmd_params = MtmdContextParams {
            use_gpu: true,
            print_timings: true,
            n_threads: 8,
            media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())?,
        };

        // Even though MtmdContext doesn't expose a lifetime in its struct def,
        // it internally handles the pointer to the model.
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

    pub fn chat<P: AsRef<Path>>(&mut self, prompt: &str, image_paths: &[P]) -> Result<String> {
        let mut result = String::new();
        let stream = self.stream_chat(prompt, image_paths)?;

        for token_res in stream {
            match token_res {
                Ok(token) => {
                    result.push_str(&token);
                }
                Err(e) => {
                    return Err(e.wrap_err(format!(
                        "Error generating token, result until now: \n{result}\n\n"
                    )));
                }
            }
        }
        Ok(result)
    }

    pub fn stream_chat<P: AsRef<Path>>(
        &mut self,
        prompt: &str,
        image_paths: &[P],
    ) -> Result<ResponseStream<'a, '_>> {
        // 1. Load all images
        let mut bitmaps = Vec::new();
        let loaded_bitmaps: Vec<MtmdBitmap> = image_paths
            .iter()
            .map(|p| {
                let path_str = p
                    .as_ref()
                    .to_str()
                    .ok_or_else(|| eyre!("Invalid path string"))?;
                MtmdBitmap::from_file(&self.mtmd_ctx, path_str)
                    .context(format!("Failed to load image: {}", path_str))
            })
            .collect::<Result<Vec<_>>>()?;

        bitmaps.extend(loaded_bitmaps.iter());

        // 2. Prepare Prompt & Apply Chat Template
        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let mut full_prompt = prompt.to_string();

        if !bitmaps.is_empty() && !full_prompt.contains(&default_marker) {
            full_prompt = format!("{} {}", default_marker, full_prompt);
        }

        let messages = vec![LlamaChatMessage::new("user".to_string(), full_prompt)?];

        let chat_template = self
            .model
            .chat_template(None)
            .context("Failed to get chat template")?;
        let formatted_prompt = self
            .model
            .apply_chat_template(&chat_template, &messages, true)
            .context("Failed to apply chat template")?;

        // 3. Tokenize & Evaluate Multimodal Input
        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = self
            .mtmd_ctx
            .tokenize(input_text, &bitmaps)
            .context("Failed to tokenize multimodal input")?;

        self.n_past = chunks
            .eval_chunks(
                &self.mtmd_ctx,
                &mut self.context,
                self.n_past as i32,
                0,
                2048,
                true,
            )
            .context("Failed to evaluate chunks")? as usize;

        // 4. Initialize Sampler
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(-1, 1.0, 0.0, 1.5),
            LlamaSampler::top_p(0.8, 1),
            LlamaSampler::temp(0.7),
            LlamaSampler::min_p(0.0, 1),
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

// -------------------------------------------------------------------------
// 2. The Iterator (Stream)
// -------------------------------------------------------------------------
pub struct ResponseStream<'a, 'b> {
    model: &'a LlamaModel,
    context: &'b mut LlamaContext<'a>,
    sampler: LlamaSampler,
    batch: LlamaBatch<'a>,
    n_past: &'b mut usize,
    is_done: bool,
}

impl<'a, 'b> Iterator for ResponseStream<'a, 'b> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            return None;
        }

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
        if let Err(e) = self.batch.add(token, *self.n_past as i32, &[0], true) {
            return Some(Err(eyre!(e)));
        }

        *self.n_past += 1;

        if let Err(e) = self.context.decode(&mut self.batch) {
            return Some(Err(eyre!("Failed to decode token: {}", e)));
        }

        Some(Ok(piece.to_string()))
    }
}

// -------------------------------------------------------------------------
// 3. Main
// -------------------------------------------------------------------------

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "assets/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf")]
    model: String,
    #[arg(
        long,
        default_value = "assets/qwen3vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf"
    )]
    mmproj: String,
    #[arg(long, default_value_t = 99)]
    n_gpu_layers: i32,
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();
    color_eyre::install()?;

    let args = Args::parse();

    // 1. Init Backend and Model
    let backend = LlamaBackend::init().context("Failed to init backend")?;
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    let model_params =
        LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers.try_into().unwrap_or(0));
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .context("Failed to load model")?;

    // 2. Initialize Session
    let mut session = MultimodalSession::new(&backend, &model, &args.mmproj, args.ctx_size)?;

    println!("--- Session Initialized ---");

    let img_hike = Path::new("assets/img/hike.png");
    let img_farm = Path::new("assets/img/farm.png");
    let img_torus = Path::new("assets/img/torus.png");
    let prompt = "Caption this image in one paragraph. Respond with the caption only.";

    info!("{}", session.chat(prompt, &[img_hike])?);
    info!("{}", session.chat(prompt, &[img_farm])?);
    info!("{}", session.chat(prompt, &[img_torus])?);

    info!("{}", session.chat("Where might this be?", &[img_hike])?);

    Ok(())
}
