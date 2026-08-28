fn main() {
    let startup_paths: Vec<std::path::PathBuf> = std::env::args_os()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("rgis")
            .with_inner_size([1280.0, 800.0]),
        multisampling: rgis_render::MSAA_SAMPLES as u16,
        ..Default::default()
    };

    eframe::run_native(
        "rgis",
        native_options,
        Box::new(move |cc| {
            let mut app = rgis_app::RgisApp::new(cc);
            if startup_paths.is_empty() {
                app.queue_load_sample();
            } else {
                app.queue_load_paths(startup_paths);
            }
            Ok(Box::new(app))
        }),
    )
    .expect("failed to run rgis native app");
}
