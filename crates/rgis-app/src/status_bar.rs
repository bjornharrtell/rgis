pub fn format_coordinates(lon: f64, lat: f64) -> String {
    let lon_hemi = if lon >= 0.0 { 'E' } else { 'W' };
    let lat_hemi = if lat >= 0.0 { 'N' } else { 'S' };
    format!(
        "{:.6}° {}   {:.6}° {}",
        lon.abs(),
        lon_hemi,
        lat.abs(),
        lat_hemi,
    )
}

pub fn format_scale(m_per_px: f64) -> String {
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

#[cfg(test)]
mod tests {
    use super::{format_coordinates, format_scale};

    #[test]
    fn formats_coordinates() {
        assert_eq!(
            format_coordinates(-123.5, 45.25),
            "123.500000° W   45.250000° N"
        );
    }

    #[test]
    fn formats_scale() {
        assert_eq!(format_scale(265.0), "1 : 1.0M");
    }
}
