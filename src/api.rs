#![allow(clippy::missing_errors_doc)]

use base64::{Engine as _, engine::general_purpose};
use color_eyre::Result;
use serde_json::{Value, json};
use std::path::Path;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

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
        // 1. Build the new message content
        let mut content = vec![json!({ "type": "text", "text": prompt })];

        // 2. Encode and append images if provided
        for path in image_paths {
            let bytes = std::fs::read(path)?;
            let b64 = general_purpose::STANDARD.encode(bytes);
            // todo: detect image format
            let data_url = format!("data:image/png;base64,{b64}");
            content.push(json!({
                "type": "image_url",
                "image_url": { "url": data_url }
            }));
        }

        // 3. Add user message to history
        self.messages.push(json!({
            "role": "user",
            "content": content
        }));

        // 4. Prepare the full request payload
        let payload = json!({
            "model": "", // llama-server usually ignores this or uses the loaded one
            "messages": self.messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 1024
        });

        // 5. Send request
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client.post(url).json(&payload).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            error!("API Error ({}): {}", status, error_text);
            return Err(color_eyre::eyre::eyre!("Llama-server returned error: {}", status));
        }

        let res_body: Value = response.json().await?;

        // 6. Extract response content
        let assistant_text = res_body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| color_eyre::eyre::eyre!("Unexpected response format: {:?}", res_body))?
            .to_string();

        // 7. Add assistant's response to history to support follow-ups
        self.messages.push(json!({
            "role": "assistant",
            "content": assistant_text
        }));

        Ok(assistant_text)
    }
}

pub async fn run() -> Result<()> {
    // 2. Define Image Paths
    let img_island = Path::new("assets/img/island.png");
    let img_farm = Path::new("assets/img/farm.png");
    let img_torus = Path::new("assets/img/torus.png");
    let prompt = "Caption this image in one paragraph. Respond with the caption only.";

    // 3. Initialize Session
    let mut session = ChatSession::new("http://localhost:8080");

    // 4. Run the requested workflow
    info!("Island: {}", session.chat(prompt, &[img_island]).await?);

    session.reset();
    info!("Farm: {}", session.chat(prompt, &[img_farm]).await?);

    session.reset();
    info!("Torus: {}", session.chat(prompt, &[img_torus]).await?);

    session.reset();
    info!("Island again: {}", session.chat(prompt, &[img_island]).await?);

    // Follow-up doesn't reset, so it knows about "Island again"
    info!(
        "Follow up: {}",
        session.chat("Where might this be?", &[] as &[&Path]).await?
    );

    // todo: compare 2 images
    // todo: streaming api? not needed for now

    Ok(())
}