use gpui::{Context, MouseButton, Window, div, prelude::*, px, rgb, rgba};
use rgis_core::{Color, LayerId, Project};

const SIDEBAR_WIDTH: f32 = 280.0;
const ZED_PANEL: u32 = 0x1b1b1b;
const ZED_SURFACE: u32 = 0x2a2a2a;
const ZED_BORDER: u32 = 0x343434;
const ZED_TEXT: u32 = 0xd4d4d4;
const ZED_MUTED: u32 = 0x929292;
const ZED_ACCENT: u32 = 0x8ab4f8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StyleColorTarget {
    Fill,
    Stroke,
}

pub trait LayerUi: Sized + 'static {
    fn project(&self) -> &Project;
    fn project_mut(&mut self) -> &mut Project;
    fn layers_expanded(&self) -> bool;
    fn set_layers_expanded(&mut self, expanded: bool);
    fn style_editor_layer(&self) -> Option<LayerId>;
    fn set_style_editor_layer(&mut self, layer: Option<LayerId>);
    fn style_color_target(&self) -> StyleColorTarget;
    fn set_style_color_target(&mut self, target: StyleColorTarget);
    fn add_layer(&mut self, window: &mut Window);
}

fn icon(path: &str, color: u32) -> impl IntoElement {
    let data = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="{path}" fill="none" stroke="#{color:06x}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"##
    );
    gpui::svg()
        .data(data.as_bytes())
        .size(px(14.0))
        .text_color(rgb(color))
        .flex_none()
}

fn color_to_rgba(color: Color) -> u32 {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u32;
    (channel(color.r) << 24) | (channel(color.g) << 16) | (channel(color.b) << 8) | channel(color.a)
}

fn format_color(color: Color) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.b.clamp(0.0, 1.0) * 255.0).round() as u8
    )
}

pub fn sidebar<T: LayerUi>(state: &T, cx: &mut Context<T>) -> impl IntoElement {
    let mut content = div()
        .w(px(SIDEBAR_WIDTH))
        .h_full()
        .p_2()
        .gap_1()
        .flex()
        .flex_col()
        .bg(rgb(ZED_PANEL))
        .border_r_1()
        .border_color(rgb(ZED_BORDER))
        .text_color(rgb(ZED_TEXT))
        .child(
            div()
                .h(px(30.0))
                .w_full()
                .px_2()
                .flex()
                .items_center()
                .text_xs()
                .text_color(rgb(ZED_MUTED))
                .child(
                    div()
                        .w(px(18.0))
                        .h(px(24.0))
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(if state.layers_expanded() {
                            icon("m6 9 6 6 6-6", ZED_MUTED)
                        } else {
                            icon("m9 6 6 6-6 6", ZED_MUTED)
                        }),
                )
                .child(div().text_xs().child("LAYERS"))
                .child(
                    div()
                        .ml_auto()
                        .w(px(24.0))
                        .h(px(24.0))
                        .flex()
                        .items_center()
                        .justify_center()
                        .hover(|style| style.bg(rgb(ZED_SURFACE)))
                        .child(icon("M12 5v14M5 12h14", ZED_MUTED))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _, window, cx| {
                                this.add_layer(window);
                                cx.stop_propagation();
                            }),
                        ),
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _, _, cx| {
                        this.set_layers_expanded(!this.layers_expanded());
                        cx.notify();
                    }),
                ),
        );

    if state.layers_expanded() {
        for layer in state.project().layers.iter().rev() {
            let id = layer.id;
            content = content.child(layer_row(state, id, layer.name.clone(), layer.visible, cx));
            if state.style_editor_layer() == Some(id) {
                content = content.child(style_panel(state, id, cx));
            }
        }
        content = content.child(
            div()
                .h(px(30.0))
                .w_full()
                .px_2()
                .gap_1()
                .flex()
                .items_center()
                .text_sm()
                .text_color(rgb(ZED_MUTED))
                .hover(|style| style.bg(rgb(ZED_SURFACE)))
                .child(
                    div()
                        .w(px(22.0))
                        .h(px(24.0))
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(icon(
                            if state.project().show_tiles {
                                "M3 6 9 3l6 3 6-3v15l-6 3-6-3-6 3V6Zm6-3v15m6-12v15"
                            } else {
                                "M3 3 21 21M3 6l6-3 6 3 6-3v9M3 12v9l6-3 2.2 1.1"
                            },
                            ZED_MUTED,
                        )),
                )
                .child(div().flex_1().child("OpenFreeMap"))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _, _, cx| {
                        this.project_mut().show_tiles = !this.project().show_tiles;
                        cx.notify();
                    }),
                ),
        );
    }
    content
}

fn layer_row<T: LayerUi>(
    _state: &T,
    layer_id: LayerId,
    name: String,
    visible: bool,
    cx: &mut Context<T>,
) -> impl IntoElement {
    let visibility_path = if visible {
        "M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Zm10 2.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"
    } else {
        "m3 3 18 18M10.6 6.2A10.7 10.7 0 0 1 12 6c6.5 0 10 6 10 6a18 18 0 0 1-3.2 3.8M6.2 6.3C3.4 8.3 2 12 2 12s3.5 6 10 6c1.1 0 2.1-.2 3-.5"
    };
    div()
        .h(px(30.0))
        .w_full()
        .px_2()
        .gap_1()
        .flex()
        .items_center()
        .text_sm()
        .text_color(rgb(ZED_TEXT))
        .hover(|style| style.bg(rgb(ZED_SURFACE)))
        .child(
            div()
                .w(px(22.0))
                .h(px(24.0))
                .flex()
                .items_center()
                .justify_center()
                .child(icon(visibility_path, ZED_MUTED))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        if let Some(layer) = this.project_mut().get_layer_mut(layer_id) {
                            layer.visible = !layer.visible;
                        }
                        cx.stop_propagation();
                        cx.notify();
                    }),
                ),
        )
        .child(
            div()
                .w(px(22.0))
                .h(px(24.0))
                .flex()
                .items_center()
                .justify_center()
                .child(icon(
                    "M12 2 3.5 6.5 12 11l8.5-4.5L12 2Zm-8.5 9.5L12 16l8.5-4.5M3.5 16.5 12 21l8.5-4.5",
                    ZED_ACCENT,
                )),
        )
        .child(div().flex_1().child(name))
        .child(
            div()
                .w(px(22.0))
                .h(px(24.0))
                .flex()
                .items_center()
                .justify_center()
                .child(icon(
                    "m12 3 1.2 5.8L19 10l-5.8 1.2L12 17l-1.2-5.8L5 10l5.8-1.2L12 3Zm6.5 12 .6 2.4 2.4.6-2.4.6-.6-2.4-2.4-.6 2.4-.6.6-2.4Z",
                    ZED_MUTED,
                ))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.set_style_editor_layer(
                            (this.style_editor_layer() != Some(layer_id)).then_some(layer_id),
                        );
                        this.set_style_color_target(StyleColorTarget::Fill);
                        cx.stop_propagation();
                        cx.notify();
                    }),
                ),
        )
        .child(
            div()
                .w(px(22.0))
                .h(px(24.0))
                .flex()
                .items_center()
                .justify_center()
                .child(icon("m6 6 12 12M18 6 6 18", ZED_MUTED))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.project_mut().remove_layer(layer_id);
                        cx.stop_propagation();
                        cx.notify();
                    }),
                ),
        )
}

fn color_chip<T: LayerUi>(
    _state: &T,
    layer_id: LayerId,
    target: StyleColorTarget,
    color: Color,
    selected: bool,
    cx: &mut Context<T>,
) -> impl IntoElement {
    div()
        .w(px(24.0))
        .h(px(24.0))
        .rounded_sm()
        .bg(rgba(color_to_rgba(color)))
        .border_1()
        .border_color(rgb(if selected { ZED_ACCENT } else { ZED_BORDER }))
        .hover(|style| style.border_color(rgb(ZED_TEXT)))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, _, cx| {
                this.set_style_color_target(target);
                if let Some(layer) = this.project_mut().get_layer_mut(layer_id) {
                    let alpha = match target {
                        StyleColorTarget::Fill => layer.style.fill.a,
                        StyleColorTarget::Stroke => layer.style.stroke.a,
                    };
                    let chosen = Color { a: alpha, ..color };
                    match target {
                        StyleColorTarget::Fill => layer.style.fill = chosen,
                        StyleColorTarget::Stroke => layer.style.stroke = chosen,
                    }
                }
                cx.stop_propagation();
                cx.notify();
            }),
        )
}

fn style_panel<T: LayerUi>(state: &T, layer_id: LayerId, cx: &mut Context<T>) -> impl IntoElement {
    let Some(layer) = state
        .project()
        .layers
        .iter()
        .find(|layer| layer.id == layer_id)
    else {
        return div();
    };
    let fill = layer.style.fill;
    let stroke = layer.style.stroke;
    let stroke_width = layer.style.stroke_width;
    let point_radius = layer.style.point_radius;
    let target = state.style_color_target();
    let target_color = match target {
        StyleColorTarget::Fill => fill,
        StyleColorTarget::Stroke => stroke,
    };
    let palette_colors = [
        (243, 139, 168),
        (250, 179, 135),
        (249, 226, 175),
        (166, 227, 161),
        (148, 226, 213),
        (137, 220, 235),
        (137, 180, 250),
        (203, 166, 247),
        (245, 194, 231),
        (205, 214, 244),
        (147, 153, 178),
        (49, 50, 68),
    ];
    let mut palette = div().flex().flex_wrap().gap_1();
    for (r, g, b) in palette_colors {
        palette = palette.child(color_chip(
            state,
            layer_id,
            target,
            Color::from_u8(r, g, b, 255),
            false,
            cx,
        ));
    }
    div()
        .w_full()
        .pl(px(48.0))
        .pr_2()
        .py_2()
        .gap_2()
        .flex()
        .flex_col()
        .bg(rgb(0x222222))
        .border_l_1()
        .border_color(rgb(ZED_BORDER))
        .text_xs()
        .text_color(rgb(ZED_MUTED))
        .child(
            div().flex().items_center().child("APPEARANCE").child(
                div()
                    .ml_auto()
                    .w(px(24.0))
                    .h(px(24.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(icon("m6 6 12 12M18 6 6 18", ZED_MUTED))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.set_style_editor_layer(None);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ),
            ),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().w(px(44.0)).child("Fill"))
                .child(color_chip(
                    state,
                    layer_id,
                    StyleColorTarget::Fill,
                    fill,
                    target == StyleColorTarget::Fill,
                    cx,
                ))
                .child(format_color(fill)),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().w(px(44.0)).child("Stroke"))
                .child(color_chip(
                    state,
                    layer_id,
                    StyleColorTarget::Stroke,
                    stroke,
                    target == StyleColorTarget::Stroke,
                    cx,
                ))
                .child(format_color(stroke)),
        )
        .child(palette)
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().w(px(84.0)).child("Opacity"))
                .child(style_button(layer_id, target, false, true, cx))
                .child(format!("{:.0}%", target_color.a * 100.0))
                .child(style_button(layer_id, target, true, true, cx)),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().w(px(84.0)).child("Line width"))
                .child(style_button(layer_id, target, false, false, cx))
                .child(format!("{stroke_width:.1}"))
                .child(style_button(layer_id, target, true, false, cx)),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().w(px(84.0)).child("Point radius"))
                .child(radius_button(layer_id, false, cx))
                .child(format!("{point_radius:.1}"))
                .child(radius_button(layer_id, true, cx)),
        )
        .child(
            div()
                .mt_1()
                .flex()
                .items_center()
                .gap_2()
                .text_color(rgb(ZED_MUTED))
                .child("Color applies to selected target")
                .child(
                    div()
                        .ml_auto()
                        .px_2()
                        .py_1()
                        .bg(rgb(ZED_SURFACE))
                        .child("Reset")
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, _, cx| {
                                if let Some(layer) = this.project_mut().get_layer_mut(layer_id) {
                                    layer.style = rgis_core::Style::default();
                                }
                                cx.stop_propagation();
                                cx.notify();
                            }),
                        ),
                ),
        )
}

fn style_button<T: LayerUi>(
    layer_id: LayerId,
    target: StyleColorTarget,
    increase: bool,
    opacity: bool,
    cx: &mut Context<T>,
) -> impl IntoElement {
    div()
        .w(px(24.0))
        .h(px(24.0))
        .flex()
        .items_center()
        .justify_center()
        .bg(rgb(ZED_SURFACE))
        .child(icon(
            if increase {
                "M12 5v14M5 12h14"
            } else {
                "M5 12h14"
            },
            ZED_MUTED,
        ))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, _, cx| {
                if let Some(layer) = this.project_mut().get_layer_mut(layer_id) {
                    if opacity {
                        let color = match target {
                            StyleColorTarget::Fill => &mut layer.style.fill,
                            StyleColorTarget::Stroke => &mut layer.style.stroke,
                        };
                        color.a = if increase {
                            (color.a + 0.05).min(1.0)
                        } else {
                            (color.a - 0.05).max(0.0)
                        };
                    } else {
                        layer.style.stroke_width = if increase {
                            layer.style.stroke_width + 0.5
                        } else {
                            (layer.style.stroke_width - 0.5).max(0.1)
                        };
                    }
                }
                cx.stop_propagation();
                cx.notify();
            }),
        )
}

fn radius_button<T: LayerUi>(
    layer_id: LayerId,
    increase: bool,
    cx: &mut Context<T>,
) -> impl IntoElement {
    div()
        .w(px(24.0))
        .h(px(24.0))
        .flex()
        .items_center()
        .justify_center()
        .bg(rgb(ZED_SURFACE))
        .child(icon(
            if increase {
                "M12 5v14M5 12h14"
            } else {
                "M5 12h14"
            },
            ZED_MUTED,
        ))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, _, cx| {
                if let Some(layer) = this.project_mut().get_layer_mut(layer_id) {
                    layer.style.point_radius = if increase {
                        layer.style.point_radius + 1.0
                    } else {
                        (layer.style.point_radius - 1.0).max(1.0)
                    };
                }
                cx.stop_propagation();
                cx.notify();
            }),
        )
}
