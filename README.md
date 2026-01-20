# qwen-llm-rs

This is a Rust implementation for running Qwen3-VL models using llama.cpp bindings. It provides basic structs for loading models and running chat sessions with image support.

## Features

- Runs chat prompts with image inputs.
- Uses llama-cpp-rs with CUDA acceleration and Flash Attention.
- Includes session support for maintaining chat history.
- Supports streaming responses.
- Option to suppress llama.cpp log output.

## Prerequisites

- Rust
- CUDA Toolkit
- Qwen3-VL GGUF and mmproj files

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/RuurdBijlsma/qwen-llm-rs.git
    cd qwen-llm-rs
    ```

2.  Download Models:
    Download the model and mmproj files from [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main) and place them in `assets/qwen3vl/`.
    
    Files used in the code by default:
    - `Qwen3VL-4B-Instruct-Q4_K_M.gguf`
    - `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`

    You can update these filenames in `src/main.rs`.

3.  Run the project:
    ```bash
    cargo run --release
    ```

## Usage Example

```rust
use qwen_llm_rs::MultimodalModel;
use std::path::Path;
use std::io::{self, Write};

fn main() -> color_eyre::Result<()> {
    let model_manager = MultimodalModel::load()?;
    let mut session = model_manager.new_session()?;

    let prompt = "What is in this image?";
    let image_path = Path::new("assets/img/island.png");

    let response = session.chat(prompt, &[image_path])?;
    println!("Model: {}", response);

    let follow_up = session.chat("Can you describe the colors?", &[] as &[&str])?;
    println!("Model: {}", follow_up);

    // Or stream the response token by token
    for token in session.stream_chat("Describe the scene.", &[] as &[&str])? {
        print!("{}", token?);
        io::stdout().flush()?;
    }

    Ok(())
}
```

## Architecture

- `MultimodalModel`: Handles backend and model initialization.
- `Session`: Manages the context, vision context, and KV cache.
- `ResponseStream`: Iterator for token generation.

## Configuration

Settings in `src/main.rs`:
- `GPU_LAYERS`: Layers to offload to GPU (default: 99).
- `CTX_SIZE`: Context window size (default: 2048).
- `SHOW_LLAMA_LOGS`: Toggle raw logs.

## License

MIT
