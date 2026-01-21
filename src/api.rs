#![allow(clippy::missing_errors_doc)]

use base64::{engine::general_purpose, Engine as _};
use color_eyre::Result;
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;
use tracing::{error, info};

pub struct ChatSession {
    base_url: String,
    client: reqwest::Client,
    messages: Vec<Value>,
}

impl ChatSession {
    #[must_use]
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
            messages: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.messages.clear();
    }

    pub async fn chat(&mut self, prompt: &str, image_paths: &[impl AsRef<Path>]) -> Result<String> {
        let mut content = vec![json!({ "type": "text", "text": prompt })];
        for path in image_paths {
            let bytes = std::fs::read(path)?;
            let mime_type = infer::get(&bytes)
                .map(|kind| kind.mime_type())
                .unwrap_or("image/jpeg");
            let b64 = general_purpose::STANDARD.encode(&bytes);
            let data_url = format!("data:{mime_type};base64,{b64}");
            content.push(json!({
                "type": "image_url",
                "image_url": { "url": data_url }
            }));
        }
        self.messages.push(json!({
            "role": "user",
            "content": content
        }));
        let payload = json!({
            "model": "",
            "messages": self.messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 1024
        });
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client.post(url).json(&payload).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            error!("API Error ({}): {}", status, error_text);
            return Err(color_eyre::eyre::eyre!(
                "Llama-server returned error: {}",
                status
            ));
        }
        let res_body: Value = response.json().await?;
        let assistant_text = res_body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| color_eyre::eyre::eyre!("Unexpected response format: {:?}", res_body))?
            .to_string();
        self.messages.push(json!({
            "role": "assistant",
            "content": assistant_text
        }));
        Ok(assistant_text)
    }
}

pub async fn run() -> Result<()> {
    // todo: add model load/unload functions to ChatSession
    // https://huggingface.co/blog/ggml-org/model-management-in-llamacpp#manually-load-a-model

    let mut session = ChatSession::new("http://localhost:8080");

    let img_island = Path::new("assets/img/island.png");
    let img_farm = Path::new("assets/img/farm.png");
    let img_torus = Path::new("assets/img/torus.png");
    let prompt = "Caption this image in one paragraph. Respond with the caption only.";

    let now = Instant::now();

    // Warmup
    info!("Island: {}", session.chat(prompt, &[img_island]).await?);
    session.reset();

    let now2 = Instant::now();

    // Actual requests
    info!("Farm: {}", session.chat(prompt, &[img_farm]).await?);
    session.reset();
    info!("Torus: {}", session.chat(prompt, &[img_torus]).await?);
    session.reset();
    info!(
        "Island again: {}",
        session.chat(prompt, &[img_island]).await?
    );
    info!(
        "Follow up: {}",
        session
            .chat("Where might this be?", &[] as &[&Path])
            .await?
    );
    info!(
        "Similarities: {}",
        session
            .chat("What are the similarities with this picture?", &[img_torus])
            .await?
    );

    info!("Total time for [API]: {:?}", now.elapsed());
    info!(
        "Total time for [API] (excluding warmup): {:?}",
        now2.elapsed()
    );

    Ok(())
}
