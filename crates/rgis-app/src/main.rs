mod drag_zoom_box;
mod map_area;
mod sidebar;
mod window;

use glib::ExitCode;
use libadwaita::prelude::*;
use libadwaita::Application;

const APP_ID: &str = "rs.rgis.app";

fn main() {
    libadwaita::init().expect("failed to initialise libadwaita");

    // Spawn the tokio runtime on a dedicated OS thread and keep it alive
    // for the lifetime of the process.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    let _rt_guard = rt.enter();
    std::mem::forget(rt);

    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    app.connect_activate(|app| {
        let win = window::RgisWindow::new(app);
        win.present();
    });

    let exit_code: ExitCode = app.run();
    std::process::exit(exit_code.get() as i32);
}
