use color_eyre::eyre::{eyre, Context, Result};
use gag::Gag;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{mtmd, send_logs_to_tracing, LogOptions};
use std::ffi::CString;
use std::io;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::Path;
use std::time::Instant;
use tracing::info;

// todo:
// * make the constants arguments for the Model struct via `bon`
// * test speed of this compared to calling llama-server via rust
const SHOW_LLAMA_LOGS: bool = false;
const MODEL_PATH: &str = "assets/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf";
const MMPROJ_PATH: &str = "assets/qwen3vl/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf";
const GPU_LAYERS: u32 = 99;
const CTX_SIZE: u32 = 2048;

pub struct MultimodalModel {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl MultimodalModel {
    pub fn load() -> Result<Self> {
        let _gags = if SHOW_LLAMA_LOGS {
            None
        } else {
            Some((
                Gag::stdout().map_err(|_| eyre!("Failed to gag stdout"))?,
                Gag::stderr().map_err(|_| eyre!("Failed to gag stderr"))?,
            ))
        };
        let backend = LlamaBackend::init().context("Failed to init backend")?;
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(SHOW_LLAMA_LOGS));
        let model_params = LlamaModelParams::default().with_n_gpu_layers(GPU_LAYERS);
        let model = LlamaModel::load_from_file(&backend, MODEL_PATH, &model_params)
            .context("Failed to load model")?;

        Ok(Self { backend, model })
    }

    pub fn new_session(&self) -> Result<Session<'_>> {
        Session::new(&self.backend, &self.model)
    }
}

pub struct Session<'a> {
    model: &'a LlamaModel,
    context: LlamaContext<'a>,
    mtmd_ctx: MtmdContext,
    batch: LlamaBatch<'a>,
    n_past: i32,
}

impl<'a> Session<'a> {
    fn new(backend: &'a LlamaBackend, model: &'a LlamaModel) -> Result<Self> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CTX_SIZE))
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            .with_n_threads(8)
            .with_n_batch(CTX_SIZE)
            .with_n_ubatch(CTX_SIZE);
        let context = model.new_context(backend, ctx_params)?;
        let mtmd_params = MtmdContextParams {
            use_gpu: true,
            n_threads: 8,
            media_marker: CString::new(mtmd::mtmd_default_marker().to_string())?,
            ..Default::default()
        };
        let mtmd_ctx = MtmdContext::init_from_file(MMPROJ_PATH, model, &mtmd_params)?;

        Ok(Self {
            model,
            context,
            mtmd_ctx,
            batch: LlamaBatch::new(CTX_SIZE as usize, 1),
            n_past: 0,
        })
    }

    pub fn reset(&mut self) {
        self.context.clear_kv_cache();
        self.n_past = 0;
    }

    pub fn chat(&mut self, prompt: &str, images: &[impl AsRef<Path>]) -> Result<String> {
        self.stream_chat(prompt, images)?.collect()
    }

    pub fn stream_chat(
        &mut self,
        prompt: &str,
        images: &[impl AsRef<Path>],
    ) -> Result<ResponseStream<'a, '_>> {
        let mut bitmaps = Vec::new();
        for p in images {
            let path_str = p.as_ref().to_str().ok_or_else(|| eyre!("Invalid path"))?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd_ctx, path_str)?);
        }
        let marker = mtmd::mtmd_default_marker().to_string();
        let full_prompt = if !bitmaps.is_empty() && !prompt.contains(&marker) {
            format!("{marker} {prompt}")
        } else {
            prompt.to_string()
        };
        let messages = vec![LlamaChatMessage::new("user".to_string(), full_prompt)?];
        let chat_template = self.model.chat_template(None)?;
        let formatted = self
            .model
            .apply_chat_template(&chat_template, &messages, true)?;
        let input = MtmdInputText {
            text: formatted,
            add_special: true,
            parse_special: true,
        };
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self.mtmd_ctx.tokenize(input, &bitmap_refs)?;
        self.n_past =
            chunks.eval_chunks(&self.mtmd_ctx, &self.context, self.n_past, 0, 4096, true)?;
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(-1, 1.0, 0.0, 1.5),
            LlamaSampler::top_p(0.8, 1),
            LlamaSampler::temp(0.7),
            LlamaSampler::greedy(),
        ]);

        Ok(ResponseStream {
            session: self,
            sampler,
            is_done: false,
        })
    }
}

pub struct ResponseStream<'a, 'b> {
    session: &'b mut Session<'a>,
    sampler: LlamaSampler,
    is_done: bool,
}

impl Iterator for ResponseStream<'_, '_> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            return None;
        }
        let token = self.sampler.sample(&self.session.context, -1);
        self.sampler.accept(token);
        if self.session.model.is_eog_token(token) {
            self.is_done = true;
            return None;
        }
        let piece = match self.session.model.token_to_str(token, Special::Tokenize) {
            Ok(s) => s,
            Err(e) => return Some(Err(eyre!(e))),
        };
        self.session.batch.clear();
        if let Err(e) = self
            .session
            .batch
            .add(token, self.session.n_past, &[0], true)
        {
            return Some(Err(eyre!(e)));
        }
        self.session.n_past += 1;
        if let Err(e) = self.session.context.decode(&mut self.session.batch) {
            return Some(Err(eyre!("Decode failed: {e}")));
        }

        Some(Ok(piece))
    }
}

pub fn run() -> Result<()> {
    let model_manager = MultimodalModel::load()?;
    let mut session = model_manager.new_session()?;

    let img_island = Path::new("assets/img/island.png");
    let img_farm = Path::new("assets/img/farm.png");
    let img_torus = Path::new("assets/img/torus.png");
    let prompt = "Caption this image in one paragraph. Respond with the caption only.";

    let now = Instant::now();

    // Warmup
    info!("Island: {}", session.chat(prompt, &[img_island])?);
    session.reset();

    let now2 = Instant::now();

    // Actual requests
    info!("Farm: {}", session.chat(prompt, &[img_farm])?);
    session.reset();
    info!("Torus: {}", session.chat(prompt, &[img_torus])?);
    session.reset();
    info!("Island again: {}", session.chat(prompt, &[img_island])?);
    info!(
        "Follow up: {}",
        session.chat("Where might this be?", &[] as &[&Path])?
    );

    info!("Total time for [bindings]: {:?}", now.elapsed());
    info!("Total time for [bindings] (excluding warmup): {:?}", now2.elapsed());

    Ok(())
}
