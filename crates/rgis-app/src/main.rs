mod drag_zoom_box;
mod map_area;
mod sidebar;
mod status_bar;
mod window;

use std::cell::RefCell;
use std::rc::Rc;

use gio::ApplicationFlags;
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
        .flags(ApplicationFlags::HANDLES_OPEN)
        .build();

    // Shared handle to the main window so both `activate` and `open` can
    // reach it (GApplication invokes one or the other depending on whether
    // file arguments were passed on the command line).
    let window: Rc<RefCell<Option<window::RgisWindow>>> = Rc::new(RefCell::new(None));

    app.connect_activate({
        let window = Rc::clone(&window);
        move |app| {
            let mut slot = window.borrow_mut();
            if slot.is_none() {
                *slot = Some(window::RgisWindow::new(app));
            }
            slot.as_ref().unwrap().present();
        }
    });

    app.connect_open({
        let window = Rc::clone(&window);
        move |app, files, _hint| {
            let mut slot = window.borrow_mut();
            if slot.is_none() {
                *slot = Some(window::RgisWindow::new(app));
            }
            let win = slot.as_ref().unwrap();
            win.present();
            for file in files {
                if let Some(path) = file.path() {
                    win.load_path(path);
                }
            }
        }
    });

    let exit_code: ExitCode = app.run();
    std::process::exit(exit_code.get() as i32);
}
