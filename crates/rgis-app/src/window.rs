use crate::map_area::MapArea;
use crate::sidebar::Sidebar;

use std::{
    cell::RefCell,
    path::PathBuf,
    rc::Rc,
    sync::Arc,
};

use gio::prelude::*;
use glib::clone;
use gtk4::prelude::*;
use libadwaita::{Application, ApplicationWindow, HeaderBar, OverlaySplitView, ToastOverlay};
use rgis_core::{Layer, Project};
use rgis_io::load;
use rgis_tiles::{OsmTileSource, TileFetcher};

pub struct RgisWindow {
    window: ApplicationWindow,
}

impl RgisWindow {
    pub fn new(app: &Application) -> Self {
        let project = Rc::new(RefCell::new(Project::default()));
        let tile_fetcher = Arc::new(TileFetcher::new(OsmTileSource));

        // ── Map canvas ────────────────────────────────────────────────────────
        let map_area = MapArea::new(Rc::clone(&project), Arc::clone(&tile_fetcher));

        // ── Sidebar ───────────────────────────────────────────────────────────
        let sidebar = Sidebar::new(Rc::clone(&project), {
            let map_area = map_area.clone();
            move || map_area.queue_render()
        });

        // ── Toast overlay ─────────────────────────────────────────────────────
        let toast_overlay = ToastOverlay::new();
        toast_overlay.set_child(Some(map_area.widget()));

        // ── Split view (sidebar + map) ─────────────────────────────────────────
        let split_view = OverlaySplitView::new();
        split_view.set_sidebar(Some(sidebar.widget()));
        split_view.set_content(Some(&toast_overlay));
        split_view.set_sidebar_width_fraction(0.28);

        // ── Header bar ────────────────────────────────────────────────────────
        let header = HeaderBar::new();

        let open_btn = gtk4::Button::with_label("Add Layer…");
        open_btn.set_icon_name("document-open-symbolic");
        header.pack_start(&open_btn);

        let toggle_sidebar_btn = gtk4::ToggleButton::new();
        toggle_sidebar_btn.set_icon_name("sidebar-show-symbolic");
        toggle_sidebar_btn.set_active(true);
        header.pack_end(&toggle_sidebar_btn);

        // ── Toolbar view ──────────────────────────────────────────────────────
        let toolbar_view = libadwaita::ToolbarView::new();
        toolbar_view.add_top_bar(&header);
        toolbar_view.set_content(Some(&split_view));

        // ── Window ────────────────────────────────────────────────────────────
        let window = ApplicationWindow::builder()
            .application(app)
            .title("rgis")
            .default_width(1280)
            .default_height(800)
            .content(&toolbar_view)
            .build();

        // Toggle sidebar
        toggle_sidebar_btn.connect_toggled(clone!(#[weak] split_view, move |btn| {
            split_view.set_show_sidebar(btn.is_active());
        }));

        // Open file button
        {
            let project = Rc::clone(&project);
            let sidebar = sidebar.clone();
            let map_area = map_area.clone();
            let toast_overlay = toast_overlay.clone();
            let window = window.clone();
            open_btn.connect_clicked(move |_| {
                open_file_dialog(
                    &window,
                    Rc::clone(&project),
                    sidebar.clone(),
                    map_area.clone(),
                    toast_overlay.clone(),
                );
            });
        }

        // Wire tile-ready notifications back to the map area.
        {
            let receiver = tile_fetcher.receiver.clone();
            let map_area = map_area.clone();
            glib::spawn_future_local(async move {
                while let Ok(ready) = receiver.recv().await {
                    map_area.on_tile_ready(ready);
                }
            });
        }

        Self { window }
    }

    pub fn present(&self) {
        self.window.present();
    }
}

// ── File open dialog ──────────────────────────────────────────────────────────

fn open_file_dialog(
    parent: &ApplicationWindow,
    project: Rc<RefCell<Project>>,
    sidebar: Sidebar,
    map_area: MapArea,
    toast_overlay: ToastOverlay,
) {
    let filter = gtk4::FileFilter::new();
    filter.set_name(Some("GIS files"));
    filter.add_pattern("*.geojson");
    filter.add_pattern("*.json");
    filter.add_pattern("*.shp");
    filter.add_pattern("*.fgb");

    let filters = gio::ListStore::new::<gtk4::FileFilter>();
    filters.append(&filter);

    let dialog = gtk4::FileDialog::builder()
        .title("Add Layer")
        .filters(&filters)
        .build();

    dialog.open(Some(parent), gio::Cancellable::NONE, move |result| {
        let Ok(file) = result else { return };
        let Some(path) = file.path() else { return };
        load_path(path, Rc::clone(&project), sidebar.clone(), map_area.clone(), toast_overlay.clone());
    });
}

fn load_path(
    path: PathBuf,
    project: Rc<RefCell<Project>>,
    sidebar: Sidebar,
    map_area: MapArea,
    toast_overlay: ToastOverlay,
) {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layer")
        .to_owned();

    glib::spawn_future_local(async move {
        match load(&path).await {
            Ok(loaded) => {
                let layer_id = {
                    let mut proj = project.borrow_mut();
                    let id = proj.next_layer_id();
                    let mut layer = Layer::new(id, loaded.name, loaded.features);
                    layer.source_path = Some(path);
                    proj.add_layer(layer);
                    id
                };
                // Zoom viewport to fit the newly loaded layer.
                let bounds = project.borrow().layers.iter()
                    .find(|l| l.id == layer_id)
                    .and_then(|l| l.bounds);
                if let Some(bounds) = bounds {
                    project.borrow_mut().viewport.fit_bounds(&bounds);
                }
                sidebar.refresh();
                map_area.invalidate_layer(layer_id);
                map_area.queue_render();
            }
            Err(e) => {
                let toast = libadwaita::Toast::new(&format!("Failed to load {name}: {e}"));
                toast_overlay.add_toast(toast);
            }
        }
    });
}
