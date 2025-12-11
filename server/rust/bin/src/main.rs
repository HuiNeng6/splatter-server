use anyhow::Result;
use posemesh_compute_node::engine::RunnerRegistry;
use posemesh_compute_node::telemetry;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env from CWD and crate dir for convenience.
    let _ = dotenvy::from_filename(".env");
    let _ = dotenvy::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/.env"));

    telemetry::init_from_env()?;

    let app = posemesh_compute_node::http::router();
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    let addr = listener.local_addr()?;
    println!("http listening on {}", addr);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let cfg = posemesh_compute_node::config::NodeConfig::from_env()?;

    let registry: RunnerRegistry = splatter_runner::registry();
    let capabilities = registry.capabilities();

    posemesh_compute_node::dds::register::spawn_registration_if_configured(&cfg, &capabilities)?;
    info!(?capabilities, "splatter runner registered capabilities");

    posemesh_compute_node::engine::run_node(cfg, registry).await?;

    Ok(())
}
