use gtk4::prelude::*;

/// Thin bar shown at the bottom of the window displaying cursor coordinates,
/// map scale and the display coordinate reference system.
#[derive(Clone)]
pub struct StatusBar {
    bar: gtk4::Box,
    coord_label: gtk4::Label,
    scale_label: gtk4::Label,
}

impl StatusBar {
    pub fn new() -> Self {
        let bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        bar.add_css_class("toolbar");

        let coord_label = gtk4::Label::new(Some("—"));
        coord_label.set_hexpand(true);
        coord_label.set_halign(gtk4::Align::Start);
        coord_label.set_margin_start(8);
        coord_label.add_css_class("monospace");

        let sep1 = gtk4::Separator::new(gtk4::Orientation::Vertical);

        let scale_label = gtk4::Label::new(Some(""));
        scale_label.set_width_chars(14);
        scale_label.set_halign(gtk4::Align::Center);
        scale_label.set_margin_start(8);
        scale_label.set_margin_end(8);
        scale_label.add_css_class("monospace");

        let sep2 = gtk4::Separator::new(gtk4::Orientation::Vertical);

        // The display coordinates are shown in WGS 84 (EPSG:4326).
        let crs_label = gtk4::Label::new(Some("EPSG:4326"));
        crs_label.set_halign(gtk4::Align::End);
        crs_label.set_margin_start(8);
        crs_label.set_margin_end(8);

        bar.append(&coord_label);
        bar.append(&sep1);
        bar.append(&scale_label);
        bar.append(&sep2);
        bar.append(&crs_label);

        Self { bar, coord_label, scale_label }
    }

    pub fn widget(&self) -> &gtk4::Box {
        &self.bar
    }

    /// Update the coordinate and scale readouts.
    ///
    /// `lon` / `lat` are in WGS-84 degrees.
    /// `m_per_px` is the Web Mercator resolution in metres per *device* pixel.
    pub fn update(&self, lon: f64, lat: f64, m_per_px: f64) {
        let lon_hemi = if lon >= 0.0 { 'E' } else { 'W' };
        let lat_hemi = if lat >= 0.0 { 'N' } else { 'S' };
        self.coord_label.set_text(&format!(
            "{:.6}° {}   {:.6}° {}",
            lon.abs(), lon_hemi,
            lat.abs(), lat_hemi,
        ));
        self.scale_label.set_text(&format_scale(m_per_px));
    }

    /// Clear the coordinate readout (e.g. when the cursor leaves the map).
    pub fn clear_coords(&self) {
        self.coord_label.set_text("—");
        self.scale_label.set_text("");
    }
}

/// Format a map scale denominator for display.
///
/// Assumes a 96 dpi screen (CSS pixel = 1/96 inch), so 1 logical pixel
/// corresponds to 0.0254 / 96 ≈ 0.000265 m on screen.
fn format_scale(m_per_px: f64) -> String {
    // scale denominator = map metres / screen metres per pixel
    let screen_m_per_px = 0.0254 / 96.0;
    let denom = m_per_px / screen_m_per_px;
    if denom >= 1_000_000.0 {
        format!("1 : {:.1}M", denom / 1_000_000.0)
    } else if denom >= 1_000.0 {
        format!("1 : {:.0}k", denom / 1_000.0)
    } else {
        format!("1 : {:.0}", denom)
    }
}
