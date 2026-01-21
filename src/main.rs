#![allow(clippy::missing_errors_doc)]

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter};

mod api;
mod bindings;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    color_eyre::install()?;

    if false {
        bindings::run()?;
    } else {
        api::run().await?;
    }

    Ok(())
}
