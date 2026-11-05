//! Shared UI built from gpui-kit components: the sidebar (layers and data
//! sources as separate sections), the layer properties dialog, the data
//! source dialog and the status bar.

use gpui::component::button::{Button, ButtonVariants as _};
use gpui::component::checkbox::Checkbox;
use gpui::component::color_picker::{ColorPicker, ColorPickerEvent, ColorPickerState};
use gpui::component::input::{Input, InputEvent, InputState};
use gpui::component::slider::{Slider, SliderEvent, SliderState};
use gpui::component::switch::Switch;
use gpui::component::tab::{Tab, TabBar};
use gpui::component::{ActiveTheme as _, Icon, Sizable as _, WindowExt as _, h_flex, v_flex};
use gpui::{
    App, Context, Entity, Hsla, IntoElement, Render, Rgba, Subscription, WeakEntity, Window, div,
    prelude::*, px,
};
use rgis_core::{Color, DataSource, DataSourceId, Layer, LayerId, Project};

const SIDEBAR_WIDTH: f32 = 280.0;
const STATUS_HEIGHT: f32 = 28.0;

const ICON_OPEN: &str = "M4 4h16v16H4zM8 4v4h8V4";
const ICON_SAVE: &str = "M5 4h10l4 4v12H5zM8 4v5h7V4M8 20v-6h8v6";
const ICON_PLUS: &str = "M12 5v14M5 12h14";
const ICON_SETTINGS: &str = "M4 6h10M18 6h2M4 12h2M10 12h10M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4";
const ICON_CLOSE: &str = "M6 6l12 12M18 6 6 18";
const ICON_SIDEBAR: &str = "M4 5h16v14H4zM9 5v14";
const ICON_SIDEBAR_HIDDEN: &str = "M4 5h16v14H4zM7 5v14";

/// A 24px stroke icon built from raw SVG path data, tinted by the text colour.
fn icon(path: &str) -> Icon {
    let data = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="{path}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>"#
    );
    Icon::empty().data(data.as_bytes())
}

fn color_to_hsla(color: Color) -> Hsla {
    Rgba {
        r: color.r,
        g: color.g,
        b: color.b,
        a: color.a,
    }
    .into()
}

fn hsla_to_color(color: Hsla) -> Color {
    let rgba = Rgba::from(color);
    Color::rgba(rgba.r, rgba.g, rgba.b, rgba.a)
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SidebarTab {
    #[default]
    Layers,
    Sources,
}

#[derive(Clone, Copy, Default)]
pub struct LayerUiState {
    tab: SidebarTab,
}

impl LayerUiState {
    pub fn tab(&self) -> SidebarTab {
        self.tab
    }

    pub fn set_tab(&mut self, tab: SidebarTab) {
        self.tab = tab;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DataSourceFormKind {
    PostgreSQL,
    Wms,
}

pub trait LayerUi: Sized + 'static {
    fn project(&self) -> &Project;
    fn project_mut(&mut self) -> &mut Project;
    fn layer_ui_state(&self) -> &LayerUiState;
    fn layer_ui_state_mut(&mut self) -> &mut LayerUiState;
    fn sidebar_visible(&self) -> bool;
    fn set_sidebar_visible(&mut self, visible: bool);
    fn status_text(&self) -> &str;
    fn set_status(&mut self, status: String);
    fn cursor_lonlat(&self) -> Option<(f64, f64)>;
    fn add_layer(&mut self, window: &mut Window);
    fn open_project(&mut self, window: &mut Window);
    fn save_project(&mut self, window: &mut Window);
}

// ── Sidebar ──────────────────────────────────────────────────────────────

pub fn sidebar<T: LayerUi>(
    state: &T,
    _window: &mut Window,
    cx: &mut Context<T>,
) -> impl IntoElement {
    let tab = state.layer_ui_state().tab();
    let selected = match tab {
        SidebarTab::Layers => 0,
        SidebarTab::Sources => 1,
    };
    let theme = cx.theme();
    let background = theme.tokens.sidebar;
    let border = theme.tokens.sidebar_border;
    let foreground = theme.tokens.sidebar_foreground;

    let toolbar = h_flex()
        .w_full()
        .p_2()
        .gap_1()
        .child(
            Button::new("project-open")
                .ghost()
                .small()
                .icon(icon(ICON_OPEN))
                .label("Open")
                .on_click(cx.listener(|this: &mut T, _, window, _| this.open_project(window))),
        )
        .child(
            Button::new("project-save")
                .ghost()
                .small()
                .icon(icon(ICON_SAVE))
                .label("Save")
                .on_click(cx.listener(|this: &mut T, _, window, _| this.save_project(window))),
        );

    let tabs = div().w_full().px_2().pb_2().child(
        TabBar::new("sidebar-tabs")
            .segmented()
            .selected_index(selected)
            .on_click(cx.listener(|this: &mut T, index: &usize, _, cx| {
                this.layer_ui_state_mut().set_tab(if *index == 0 {
                    SidebarTab::Layers
                } else {
                    SidebarTab::Sources
                });
                cx.notify();
            }))
            .child(Tab::new().label("Layers"))
            .child(Tab::new().label("Data sources")),
    );

    let body = match tab {
        SidebarTab::Layers => layers_section(state, cx).into_any_element(),
        SidebarTab::Sources => sources_section(state, cx).into_any_element(),
    };

    v_flex()
        .w(px(SIDEBAR_WIDTH))
        .h_full()
        .flex_none()
        .bg(background)
        .border_r_1()
        .border_color(border)
        .text_color(foreground)
        .child(toolbar)
        .child(tabs)
        .child(body)
}

fn section_header<T: LayerUi>(
    title: &'static str,
    add_id: &'static str,
    add_tooltip: &'static str,
    on_add: impl Fn(&mut T, &mut Window, &mut Context<T>) + 'static,
    cx: &mut Context<T>,
) -> impl IntoElement {
    let muted = cx.theme().muted_foreground;
    h_flex()
        .w_full()
        .px_3()
        .pb_1()
        .items_center()
        .child(div().text_xs().text_color(muted).child(title))
        .child(
            div().ml_auto().child(
                Button::new(add_id)
                    .ghost()
                    .small()
                    .icon(icon(ICON_PLUS))
                    .tooltip(add_tooltip)
                    .on_click(
                        cx.listener(move |this: &mut T, _, window, cx| on_add(this, window, cx)),
                    ),
            ),
        )
}

fn layers_section<T: LayerUi>(state: &T, cx: &mut Context<T>) -> impl IntoElement {
    let muted = cx.theme().muted_foreground;
    let mut list = v_flex().w_full().px_1().gap_0p5();
    for layer in state.project().layers.iter().rev() {
        list = list.child(layer_row(layer, cx));
    }
    if state.project().layers.is_empty() {
        list = list.child(
            div()
                .px_2()
                .py_2()
                .text_xs()
                .text_color(muted)
                .child("No layers yet"),
        );
    }
    let show_tiles = state.project().show_tiles;
    let basemap = h_flex()
        .w_full()
        .px_3()
        .py_2()
        .mt_2()
        .gap_2()
        .items_center()
        .child(div().flex_1().text_sm().child("OpenFreeMap basemap"))
        .child(
            Switch::new("basemap-visible")
                .checked(show_tiles)
                .on_click(cx.listener(|this: &mut T, checked: &bool, _, cx| {
                    this.project_mut().show_tiles = *checked;
                    cx.notify();
                })),
        );
    v_flex()
        .w_full()
        .child(section_header::<T>(
            "LAYERS",
            "layer-add",
            "Add layer",
            |this, window, _| this.add_layer(window),
            cx,
        ))
        .child(list)
        .child(basemap)
}

fn layer_row<T: LayerUi>(layer: &Layer, cx: &mut Context<T>) -> impl IntoElement {
    let id = layer.id;
    let hover = cx.theme().tokens.sidebar_accent;
    h_flex()
        .id(("layer-row", id.0))
        .w_full()
        .h(px(32.0))
        .px_2()
        .gap_2()
        .items_center()
        .rounded_md()
        .hover(move |style| style.bg(hover))
        .child(
            Checkbox::new(("layer-visible", id.0))
                .checked(layer.visible)
                .on_click(cx.listener(move |this: &mut T, checked: &bool, _, cx| {
                    if let Some(layer) = this.project_mut().get_layer_mut(id) {
                        layer.visible = *checked;
                    }
                    cx.notify();
                })),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .truncate()
                .child(layer.name.clone()),
        )
        .child(
            Button::new(("layer-properties", id.0))
                .ghost()
                .small()
                .icon(icon(ICON_SETTINGS))
                .tooltip("Layer properties")
                .on_click(cx.listener(move |this: &mut T, _, window, cx| {
                    open_layer_dialog(this, id, window, cx);
                })),
        )
        .on_click(
            cx.listener(move |this: &mut T, event: &gpui::ClickEvent, window, cx| {
                if event.click_count() == 2 {
                    open_layer_dialog(this, id, window, cx);
                }
            }),
        )
}

fn sources_section<T: LayerUi>(state: &T, cx: &mut Context<T>) -> impl IntoElement {
    let muted = cx.theme().muted_foreground;
    let hover = cx.theme().tokens.sidebar_accent;
    let mut list = v_flex().w_full().px_1().gap_0p5();
    for source in state.project().data_sources.iter().rev() {
        let id: DataSourceId = source.id;
        list = list.child(
            h_flex()
                .id(("source-row", id.0))
                .w_full()
                .min_h(px(40.0))
                .px_2()
                .gap_2()
                .items_center()
                .rounded_md()
                .hover(move |style| style.bg(hover))
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .child(div().truncate().text_sm().child(source.name.clone()))
                        .child(
                            div()
                                .truncate()
                                .text_xs()
                                .text_color(muted)
                                .child(data_source_summary(source)),
                        ),
                )
                .child(
                    Button::new(("source-remove", id.0))
                        .ghost()
                        .small()
                        .icon(icon(ICON_CLOSE))
                        .tooltip("Remove data source")
                        .on_click(cx.listener(move |this: &mut T, _, _, cx| {
                            this.project_mut().remove_data_source(id);
                            cx.notify();
                        })),
                ),
        );
    }
    if state.project().data_sources.is_empty() {
        list = list.child(
            div()
                .px_2()
                .py_2()
                .text_xs()
                .text_color(muted)
                .child("No data sources yet"),
        );
    }
    v_flex()
        .w_full()
        .child(section_header::<T>(
            "DATA SOURCES",
            "source-add",
            "Add data source",
            |_, window, cx| open_data_source_dialog(cx.entity().downgrade(), window, cx),
            cx,
        ))
        .child(list)
}

fn data_source_summary(source: &DataSource) -> String {
    match &source.kind {
        rgis_core::DataSourceKind::PostgreSQL {
            host,
            port,
            database,
            ..
        } => format!("PostgreSQL · {host}:{port}/{database}"),
        rgis_core::DataSourceKind::Wms { url, layers } => format!("WMS · {url} ({layers})"),
    }
}

// ── Status bar ───────────────────────────────────────────────────────────

pub fn status_bar<T: LayerUi>(state: &T, cx: &mut Context<T>) -> impl IntoElement {
    let theme = cx.theme();
    let background = theme.tokens.sidebar;
    let border = theme.tokens.border;
    let muted = theme.muted_foreground;
    let visible = state.sidebar_visible();
    h_flex()
        .h(px(STATUS_HEIGHT))
        .flex_none()
        .items_center()
        .border_t_1()
        .border_color(border)
        .bg(background)
        .text_xs()
        .text_color(muted)
        .child(
            div()
                .h_full()
                .w(px(if visible { SIDEBAR_WIDTH } else { 36.0 }))
                .flex()
                .items_center()
                .px_1()
                .border_r_1()
                .border_color(border)
                .child(
                    Button::new("sidebar-toggle")
                        .ghost()
                        .xsmall()
                        .icon(icon(if visible {
                            ICON_SIDEBAR
                        } else {
                            ICON_SIDEBAR_HIDDEN
                        }))
                        .tooltip("Toggle sidebar")
                        .on_click(cx.listener(|this: &mut T, _, _, cx| {
                            let visible = this.sidebar_visible();
                            this.set_sidebar_visible(!visible);
                            cx.notify();
                        })),
                ),
        )
        .child(
            div()
                .h_full()
                .flex_1()
                .px_3()
                .flex()
                .items_center()
                .child(state.status_text().to_string()),
        )
        .child(
            div()
                .h_full()
                .px_3()
                .flex()
                .items_center()
                .border_l_1()
                .border_color(border)
                .child(format!("zoom {:.2}", state.project().viewport.zoom)),
        )
        .child(
            div()
                .h_full()
                .min_w(px(150.0))
                .px_3()
                .flex()
                .items_center()
                .border_l_1()
                .border_color(border)
                .child(
                    state
                        .cursor_lonlat()
                        .map(|(lon, lat)| format!("{lon:.5}, {lat:.5}"))
                        .unwrap_or_else(|| "-".to_string()),
                ),
        )
}

// ── Layer properties dialog ──────────────────────────────────────────────

fn update_layer<T: LayerUi>(
    app: &WeakEntity<T>,
    id: LayerId,
    cx: &mut App,
    change: impl FnOnce(&mut Layer),
) {
    let _ = app.update(cx, |this, cx| {
        if let Some(layer) = this.project_mut().get_layer_mut(id) {
            change(layer);
        }
        cx.notify();
    });
}

/// Opens the properties dialog for a layer. Call it from an event handler of
/// the app: the layer is read from `app` here because the app entity is already
/// being updated and can not be read again from inside the dialog.
pub fn open_layer_dialog<T: LayerUi>(
    app: &T,
    layer_id: LayerId,
    window: &mut Window,
    cx: &mut Context<T>,
) {
    let Some(layer) = app
        .project()
        .layers
        .iter()
        .find(|layer| layer.id == layer_id)
    else {
        return;
    };
    let (name, style) = (layer.name.clone(), layer.style.clone());
    let weak = cx.entity().downgrade();
    let view = cx.new(|cx| LayerProperties::new(weak, layer_id, name, style, window, cx));
    window.open_dialog(cx, move |dialog, _, _| {
        dialog
            .title("Layer properties")
            .w(px(380.0))
            .child(view.clone())
    });
}

struct LayerProperties<T: LayerUi> {
    app: WeakEntity<T>,
    layer_id: LayerId,
    name: Entity<InputState>,
    fill: Entity<ColorPickerState>,
    stroke: Entity<ColorPickerState>,
    stroke_width: Entity<SliderState>,
    point_radius: Entity<SliderState>,
    _subscriptions: Vec<Subscription>,
}

impl<T: LayerUi> LayerProperties<T> {
    fn new(
        app: WeakEntity<T>,
        layer_id: LayerId,
        name: String,
        style: rgis_core::Style,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let name_input = cx.new(|cx| InputState::new(window, cx).default_value(name));
        let fill =
            cx.new(|cx| ColorPickerState::new(window, cx).default_value(color_to_hsla(style.fill)));
        let stroke = cx
            .new(|cx| ColorPickerState::new(window, cx).default_value(color_to_hsla(style.stroke)));
        let stroke_width = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(12.0)
                .step(0.1)
                .default_value(style.stroke_width)
        });
        let point_radius = cx.new(|_| {
            SliderState::new()
                .min(1.0)
                .max(24.0)
                .step(0.5)
                .default_value(style.point_radius)
        });

        let mut subscriptions = Vec::new();
        subscriptions.push(cx.subscribe_in(
            &name_input,
            window,
            move |this: &mut Self, input, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    let name = input.read(cx).value().to_string();
                    update_layer(&this.app, this.layer_id, cx, |layer| layer.name = name);
                }
            },
        ));
        subscriptions.push(cx.subscribe_in(
            &fill,
            window,
            |this: &mut Self, _, event: &ColorPickerEvent, _, cx| {
                let ColorPickerEvent::Change(Some(color)) = event else {
                    return;
                };
                let color = hsla_to_color(*color);
                update_layer(&this.app, this.layer_id, cx, |layer| {
                    layer.style.fill = color;
                });
            },
        ));
        subscriptions.push(cx.subscribe_in(
            &stroke,
            window,
            |this: &mut Self, _, event: &ColorPickerEvent, _, cx| {
                let ColorPickerEvent::Change(Some(color)) = event else {
                    return;
                };
                let color = hsla_to_color(*color);
                update_layer(&this.app, this.layer_id, cx, |layer| {
                    layer.style.stroke = color;
                });
            },
        ));
        subscriptions.push(cx.subscribe_in(
            &stroke_width,
            window,
            |this: &mut Self, _, event: &SliderEvent, _, cx| {
                let SliderEvent::Change(value) = event else {
                    return;
                };
                let width = value.end();
                update_layer(&this.app, this.layer_id, cx, |layer| {
                    layer.style.stroke_width = width;
                });
            },
        ));
        subscriptions.push(cx.subscribe_in(
            &point_radius,
            window,
            |this: &mut Self, _, event: &SliderEvent, _, cx| {
                let SliderEvent::Change(value) = event else {
                    return;
                };
                let radius = value.end();
                update_layer(&this.app, this.layer_id, cx, |layer| {
                    layer.style.point_radius = radius;
                });
            },
        ));

        Self {
            app,
            layer_id,
            name: name_input,
            fill,
            stroke,
            stroke_width,
            point_radius,
            _subscriptions: subscriptions,
        }
    }
}

fn labelled<E: IntoElement>(label: &'static str, muted: Hsla, control: E) -> impl IntoElement {
    v_flex()
        .w_full()
        .gap_1()
        .child(div().text_xs().text_color(muted).child(label))
        .child(control)
}

impl<T: LayerUi> Render for LayerProperties<T> {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let layer_id = self.layer_id;
        let app = self.app.clone();
        let zoom_app = self.app.clone();
        let remove_app = self.app.clone();
        let width = self.stroke_width.read(cx).value().end();
        let radius = self.point_radius.read(cx).value().end();

        v_flex()
            .w_full()
            .gap_4()
            .child(labelled("Name", muted, Input::new(&self.name)))
            .child(
                h_flex()
                    .w_full()
                    .gap_4()
                    .child(div().flex_1().child(labelled(
                        "Fill",
                        muted,
                        ColorPicker::new(&self.fill).label("Fill"),
                    )))
                    .child(div().flex_1().child(labelled(
                        "Stroke",
                        muted,
                        ColorPicker::new(&self.stroke).label("Stroke"),
                    ))),
            )
            .child(labelled(
                "Stroke width",
                muted,
                h_flex()
                    .w_full()
                    .gap_3()
                    .items_center()
                    .child(div().flex_1().child(Slider::new(&self.stroke_width)))
                    .child(div().w(px(36.0)).text_sm().child(format!("{width:.1}"))),
            ))
            .child(labelled(
                "Point radius",
                muted,
                h_flex()
                    .w_full()
                    .gap_3()
                    .items_center()
                    .child(div().flex_1().child(Slider::new(&self.point_radius)))
                    .child(div().w(px(36.0)).text_sm().child(format!("{radius:.1}"))),
            ))
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .child(
                        Button::new("layer-zoom")
                            .small()
                            .label("Zoom to layer")
                            .on_click(move |_, _, cx| {
                                let _ = zoom_app.update(cx, |this, cx| {
                                    let bounds = this
                                        .project()
                                        .layers
                                        .iter()
                                        .find(|layer| layer.id == layer_id)
                                        .and_then(|layer| layer.bounds);
                                    if let Some(bounds) = bounds {
                                        this.project_mut().viewport.fit_bounds(&bounds);
                                    }
                                    cx.notify();
                                });
                            }),
                    )
                    .child(
                        Button::new("layer-reset")
                            .small()
                            .label("Reset style")
                            .on_click(move |_, window, cx| {
                                update_layer(&app, layer_id, cx, |layer| {
                                    layer.style = rgis_core::Style::default();
                                });
                                window.close_dialog(cx);
                            }),
                    )
                    .child(
                        div().ml_auto().child(
                            Button::new("layer-remove")
                                .small()
                                .danger()
                                .label("Remove")
                                .on_click(move |_, window, cx| {
                                    let _ = remove_app.update(cx, |this, cx| {
                                        this.project_mut().remove_layer(layer_id);
                                        cx.notify();
                                    });
                                    window.close_dialog(cx);
                                }),
                        ),
                    ),
            )
    }
}

// ── Data source dialog ───────────────────────────────────────────────────

pub fn open_data_source_dialog<T: LayerUi>(app: WeakEntity<T>, window: &mut Window, cx: &mut App) {
    let view = cx.new(|cx| DataSourceForm::new(app, window, cx));
    window.open_dialog(cx, move |dialog, _, _| {
        dialog
            .title("Add data source")
            .w(px(420.0))
            .child(view.clone())
    });
}

struct DataSourceForm<T: LayerUi> {
    app: WeakEntity<T>,
    kind: DataSourceFormKind,
    name: Entity<InputState>,
    host: Entity<InputState>,
    port: Entity<InputState>,
    database: Entity<InputState>,
    username: Entity<InputState>,
    password: Entity<InputState>,
    url: Entity<InputState>,
    layers: Entity<InputState>,
    error: Option<String>,
}

impl<T: LayerUi> DataSourceForm<T> {
    fn new(app: WeakEntity<T>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = |window: &mut Window, cx: &mut Context<Self>, placeholder: &'static str| {
            cx.new(|cx| InputState::new(window, cx).placeholder(placeholder))
        };
        let host = cx.new(|cx| InputState::new(window, cx).default_value("localhost"));
        let port = cx.new(|cx| InputState::new(window, cx).default_value("5432"));
        let password = cx.new(|cx| InputState::new(window, cx).masked(true));
        Self {
            app,
            kind: DataSourceFormKind::PostgreSQL,
            name: input(window, cx, "Display name"),
            host,
            port,
            database: input(window, cx, "Database"),
            username: input(window, cx, "Username"),
            password,
            url: input(window, cx, "https://example.com/wms"),
            layers: input(window, cx, "layer_a,layer_b"),
            error: None,
        }
    }

    fn text(input: &Entity<InputState>, cx: &App) -> String {
        input.read(cx).value().trim().to_owned()
    }

    fn build(&self, cx: &mut App, id: DataSourceId) -> Result<DataSource, String> {
        let name = Self::text(&self.name, cx);
        match self.kind {
            DataSourceFormKind::PostgreSQL => {
                let host = Self::text(&self.host, cx);
                let database = Self::text(&self.database, cx);
                let username = Self::text(&self.username, cx);
                let password = self.password.read(cx).value().to_string();
                let port = match Self::text(&self.port, cx).parse::<u16>() {
                    Ok(port) if port > 0 => port,
                    _ => return Err("Port must be a number between 1 and 65535".to_string()),
                };
                if name.is_empty() || host.is_empty() || database.is_empty() || username.is_empty()
                {
                    return Err("Name, host, database and username are required".to_string());
                }
                Ok(DataSource::postgresql(
                    id, name, host, port, database, username, password,
                ))
            }
            DataSourceFormKind::Wms => {
                let url = Self::text(&self.url, cx);
                let layers = Self::text(&self.layers, cx);
                if name.is_empty() || url.is_empty() || layers.is_empty() {
                    return Err("Name, service URL and layers are required".to_string());
                }
                Ok(DataSource::wms(id, name, url, layers))
            }
        }
    }
}

impl<T: LayerUi> Render for DataSourceForm<T> {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        let danger = theme.danger;
        let selected = match self.kind {
            DataSourceFormKind::PostgreSQL => 0,
            DataSourceFormKind::Wms => 1,
        };
        let mut fields =
            v_flex()
                .w_full()
                .gap_3()
                .child(labelled("Name", muted, Input::new(&self.name)));
        fields = match self.kind {
            DataSourceFormKind::PostgreSQL => fields
                .child(
                    h_flex()
                        .w_full()
                        .gap_3()
                        .child(div().flex_1().child(labelled(
                            "Host",
                            muted,
                            Input::new(&self.host),
                        )))
                        .child(div().w(px(96.0)).child(labelled(
                            "Port",
                            muted,
                            Input::new(&self.port),
                        ))),
                )
                .child(labelled("Database", muted, Input::new(&self.database)))
                .child(
                    h_flex()
                        .w_full()
                        .gap_3()
                        .child(div().flex_1().child(labelled(
                            "Username",
                            muted,
                            Input::new(&self.username),
                        )))
                        .child(div().flex_1().child(labelled(
                            "Password",
                            muted,
                            Input::new(&self.password),
                        ))),
                ),
            DataSourceFormKind::Wms => fields
                .child(labelled("Service URL", muted, Input::new(&self.url)))
                .child(labelled("Layers", muted, Input::new(&self.layers))),
        };

        v_flex()
            .w_full()
            .gap_4()
            .child(
                TabBar::new("source-kind")
                    .segmented()
                    .selected_index(selected)
                    .on_click(cx.listener(|this: &mut Self, index: &usize, _, cx| {
                        this.kind = if *index == 0 {
                            DataSourceFormKind::PostgreSQL
                        } else {
                            DataSourceFormKind::Wms
                        };
                        this.error = None;
                        cx.notify();
                    }))
                    .child(Tab::new().label("PostgreSQL"))
                    .child(Tab::new().label("WMS")),
            )
            .child(fields)
            .children(
                self.error
                    .clone()
                    .map(|error| div().text_sm().text_color(danger).child(error)),
            )
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .justify_end()
                    .child(
                        Button::new("source-cancel")
                            .label("Cancel")
                            .on_click(|_, window, cx| window.close_dialog(cx)),
                    )
                    .child(
                        Button::new("source-add-confirm")
                            .primary()
                            .label("Add data source")
                            .on_click(cx.listener(|this: &mut Self, _, window, cx| {
                                let Some(app) = this.app.upgrade() else {
                                    return;
                                };
                                let id = app
                                    .update(cx, |app, _| app.project_mut().next_data_source_id());
                                match this.build(cx, id) {
                                    Ok(source) => {
                                        let name = source.name.clone();
                                        app.update(cx, |app, cx| {
                                            app.project_mut().add_data_source(source);
                                            app.set_status(format!("Added data source {name}"));
                                            cx.notify();
                                        });
                                        window.close_dialog(cx);
                                    }
                                    Err(error) => {
                                        this.error = Some(error);
                                        cx.notify();
                                    }
                                }
                            })),
                    ),
            )
    }
}
