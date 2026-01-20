use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use anyhow::{Context, Result};
use clap::Parser;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::Special;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model GGUF file
    #[arg(long, default_value = "assets/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf")]
    model: String,

    /// Path to the mmproj GGUF file
    #[arg(long, default_value = "assets/qwen3vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf")]
    mmproj: String,

    /// Path to the image file
    #[arg(long, default_value = "assets/img/test-image.png")]
    image: String,

    /// Prompt for the model
    #[arg(long, default_value = "Generate a caption for this image in 3 sentences. Only respond with the caption.")]
    prompt: String,

    /// Number of GPU layers to offload
    #[arg(long, default_value_t = 99)]
    n_gpu_layers: i32,

    /// Context size
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Initializing Llama backend...");
    let backend = LlamaBackend::init().context("Failed to initialize Llama backend")?;

    // Suppress internal llama.cpp and ggml logs that pollute the output
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    // 1. Load Model
    println!("Loading model from {}...", args.model);
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(args.n_gpu_layers.try_into().unwrap_or(0));
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .context("Failed to load model")?;

    // 2. Create Context
    println!("Creating context with size {}...", args.ctx_size);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(args.ctx_size))
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
        .with_n_threads(8)
        .with_n_batch(2048)
        .with_n_ubatch(512);
    let mut context = model.new_context(&backend, ctx_params)
        .context("Failed to create context")?;

    // 3. Initialize MTMD Context
    println!("Initializing MTMD context with projector {}...", args.mmproj);
    let mtmd_params = MtmdContextParams {
        use_gpu: true,
        print_timings: true,
        n_threads: 8,
        media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())?,
    };
    let mtmd_ctx = MtmdContext::init_from_file(&args.mmproj, &model, &mtmd_params)
        .context("Failed to initialize MTMD context")?;

    // 4. Load Media
    println!("Loading image from {}...", args.image);
    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, &args.image)
        .context("Failed to load image")?;

    // 5. Prepare Prompt & Apply Chat Template
    let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
    let mut full_prompt = args.prompt.clone();
    if !full_prompt.contains(&default_marker) {
        full_prompt = format!("{} {}", default_marker, full_prompt);
    }

    let messages = vec![
        LlamaChatMessage::new("user".to_string(), full_prompt)?
    ];

    let chat_template = model.chat_template(None)
        .context("Failed to get chat template")?;
    let formatted_prompt = model.apply_chat_template(&chat_template, &messages, true)
        .context("Failed to apply chat template")?;

    // 6. Tokenize & Evaluate Multimodal Input
    println!("Tokenizing and evaluating input...");
    let input_text = MtmdInputText {
        text: formatted_prompt,
        add_special: true,
        parse_special: true,
    };
    let bitmaps = vec![&bitmap];
    let chunks = mtmd_ctx.tokenize(input_text, &bitmaps)
        .context("Failed to tokenize multimodal input")?;

    let mut n_past = chunks.eval_chunks(&mtmd_ctx, &mut context, 0, 0, 2048, true)
        .context("Failed to evaluate chunks")?;

    // 7. Sampler Setup
    // --top-p 0.8 --temp 0.7 --min-p 0.0 --presence-penalty 1.5
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(-1, 1.0, 0.0, 1.5),
        LlamaSampler::top_p(0.8, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::min_p(0.0, 1),
        LlamaSampler::greedy(),
    ]);

    // 8. Generation Loop
    println!("\nGenerating response...\n");
    let mut batch = LlamaBatch::new(args.ctx_size as usize, 1);
    
    loop {
        let token = sampler.sample(&context, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            println!();
            break;
        }

        let piece = model.token_to_str(token, Special::Tokenize)?;
        print!("{}", piece);
        io::stdout().flush()?;

        batch.clear();
        batch.add(token, n_past, &[0], true)?;
        n_past += 1;
        context.decode(&mut batch).context("Failed to decode token")?;
    }

    println!("\nInference complete.");

    Ok(())
}
