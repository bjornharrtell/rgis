//! Chromeless window header modelled on Zed's title bar: a hamburger button
//! that expands into the application menu bar while a menu is open, then
//! collapses again. Project open/save and the other app commands live here.

use gpui::component::button::{Button, ButtonVariants as _};
use gpui::component::menu::{PopupMenu, PopupMenuItem};
use gpui::component::popover::Popover;
use gpui::component::{ActiveTheme as _, Icon, Sizable as _, TitleBar, h_flex};
use gpui::prelude::FluentBuilder as _;
use gpui::{
    Context, DismissEvent, Entity, Focusable as _, InteractiveElement as _, IntoElement,
    ParentElement as _, Render, StatefulInteractiveElement as _, Styled as _, Subscription,
    WeakEntity, Window, div, px,
};

use crate::ui::{LayerUi, SidebarTab, open_data_source_dialog};

const ICON_MENU: &str = "M4 6h16M4 12h16M4 18h16";

fn icon(path: &str) -> Icon {
    let data = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="{path}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>"#
    );
    Icon::empty().data(data.as_bytes())
}

const MENU_NAMES: [&str; 2] = ["File", "View"];

#[derive(Clone, Copy)]
enum Cmd {
    OpenProject,
    SaveProject,
    AddLayer,
    AddDataSource,
    Quit,
    ToggleSidebar,
    ToggleBasemap,
    ShowLayers,
    ShowSources,
    ZoomToLayers,
}

/// The entries of a menu; `None` is a separator.
fn menu_items(index: usize) -> Vec<Option<(&'static str, Cmd)>> {
    match index {
        0 => vec![
            Some(("Open Project…", Cmd::OpenProject)),
            Some(("Save Project", Cmd::SaveProject)),
            None,
            Some(("Add Layer…", Cmd::AddLayer)),
            Some(("Add Data Source…", Cmd::AddDataSource)),
            None,
            Some(("Quit", Cmd::Quit)),
        ],
        _ => vec![
            Some(("Toggle Sidebar", Cmd::ToggleSidebar)),
            Some(("Show Basemap", Cmd::ToggleBasemap)),
            None,
            Some(("Layers", Cmd::ShowLayers)),
            Some(("Data Sources", Cmd::ShowSources)),
            None,
            Some(("Zoom to Layers", Cmd::ZoomToLayers)),
        ],
    }
}

fn run_command<T: LayerUi>(command: Cmd, app: &mut T, window: &mut Window, cx: &mut Context<T>) {
    match command {
        Cmd::OpenProject => app.open_project(window),
        Cmd::SaveProject => app.save_project(window),
        Cmd::AddLayer => app.add_layer(window),
        Cmd::AddDataSource => {
            app.set_sidebar_visible(true);
            app.layer_ui_state_mut().set_tab(SidebarTab::Sources);
            open_data_source_dialog(cx.entity().downgrade(), window, cx);
        }
        Cmd::Quit => cx.quit(),
        Cmd::ToggleSidebar => {
            let visible = app.sidebar_visible();
            app.set_sidebar_visible(!visible);
        }
        Cmd::ToggleBasemap => {
            let visible = app.project().show_tiles;
            app.project_mut().show_tiles = !visible;
        }
        Cmd::ShowLayers => {
            app.set_sidebar_visible(true);
            app.layer_ui_state_mut().set_tab(SidebarTab::Layers);
        }
        Cmd::ShowSources => {
            app.set_sidebar_visible(true);
            app.layer_ui_state_mut().set_tab(SidebarTab::Sources);
        }
        Cmd::ZoomToLayers => {
            let bounds = app
                .project()
                .layers
                .iter()
                .filter(|layer| layer.visible)
                .filter_map(|layer| layer.bounds)
                .reduce(|left, right| left.union(&right));
            if let Some(bounds) = bounds {
                app.project_mut().viewport.fit_bounds(&bounds);
            }
        }
    }
    cx.notify();
}

pub struct AppHeader<T: LayerUi> {
    app: WeakEntity<T>,
    /// The menu currently open. While one is open the hamburger is replaced by
    /// the full menu bar.
    open: Option<usize>,
    popup: Option<Entity<PopupMenu>>,
    _popup_subscription: Option<Subscription>,
}

impl<T: LayerUi> AppHeader<T> {
    fn new(app: WeakEntity<T>) -> Self {
        Self {
            app,
            open: None,
            popup: None,
            _popup_subscription: None,
        }
    }

    fn close(&mut self, cx: &mut Context<Self>) {
        self.open = None;
        self.popup = None;
        self._popup_subscription = None;
        cx.notify();
    }

    fn open_menu(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let app = self.app.clone();
        let basemap_checked = app
            .read_with(cx, |this, _| this.project().show_tiles)
            .unwrap_or(true);
        let popup = PopupMenu::build(window, cx, move |mut menu, _, _| {
            for entry in menu_items(index) {
                menu = match entry {
                    None => menu.separator(),
                    Some((label, command)) => {
                        let app = app.clone();
                        let mut item = PopupMenuItem::new(label);
                        if matches!(command, Cmd::ToggleBasemap) {
                            item = item.checked(basemap_checked);
                        }
                        menu.item(item.on_click(move |_, window, cx| {
                            let _ = app.update(cx, |this, cx| {
                                run_command(command, this, window, cx);
                            });
                        }))
                    }
                };
            }
            menu
        });
        popup.focus_handle(cx).focus(window, cx);
        self._popup_subscription = Some(cx.subscribe_in(
            &popup,
            window,
            |this, _, _: &DismissEvent, _, cx| this.close(cx),
        ));
        self.popup = Some(popup);
        self.open = Some(index);
        cx.notify();
    }

    fn menu_trigger(&self, index: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        let header = cx.entity().downgrade();
        let popup = self.popup.clone();
        let is_open = self.open == Some(index);
        div()
            .id(("app-menu", index))
            .on_hover(cx.listener(move |this, hovered: &bool, window, cx| {
                // Hovering a sibling while a menu is open switches to it.
                if *hovered && this.open.is_some() && this.open != Some(index) {
                    this.open_menu(index, window, cx);
                }
            }))
            .child(
                Popover::new(("app-menu-popover", index))
                    .appearance(false)
                    .overlay_closable(false)
                    .open(is_open)
                    .on_open_change(move |open, _, cx| {
                        if !*open {
                            let _ = header.update(cx, |this, cx| {
                                if this.open == Some(index) {
                                    this.close(cx);
                                }
                            });
                        }
                    })
                    .trigger(
                        Button::new(("app-menu-trigger", index))
                            .ghost()
                            .small()
                            .label(MENU_NAMES[index])
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.open_menu(index, window, cx);
                            })),
                    )
                    .content(move |_, _, _| {
                        popup
                            .clone()
                            .map(|popup| popup.into_any_element())
                            .unwrap_or_else(|| div().into_any_element())
                    }),
            )
            .into_any_element()
    }
}

impl<T: LayerUi> Render for AppHeader<T> {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let background = theme.title_bar;
        let muted = theme.muted_foreground;
        let expanded = self.open.is_some();

        let menu = if expanded {
            h_flex()
                .gap_0p5()
                .children(
                    (0..MENU_NAMES.len())
                        .map(|index| self.menu_trigger(index, cx))
                        .collect::<Vec<_>>(),
                )
                .into_any_element()
        } else {
            Button::new("app-menu-hamburger")
                .ghost()
                .small()
                .icon(icon(ICON_MENU))
                .tooltip("Open Application Menu")
                .on_click(cx.listener(|this, _, window, cx| this.open_menu(0, window, cx)))
                .into_any_element()
        };

        TitleBar::new()
            .bg(background)
            .border_color(background)
            .pl(px(6.0))
            .child(
                h_flex()
                    .h_full()
                    .gap_2()
                    .items_center()
                    .child(menu)
                    // Like Zed, the project label gives way to the expanded menu bar.
                    .when(!expanded, |this| {
                        this.child(div().text_sm().text_color(muted).child("rgis"))
                    }),
            )
    }
}

/// The header entity for the current window. Create it from the app's render.
pub fn app_header<T: LayerUi>(window: &mut Window, cx: &mut Context<T>) -> Entity<AppHeader<T>> {
    let app = cx.entity().downgrade();
    window.use_keyed_state("app-header", cx, move |_, _| AppHeader::new(app))
}
