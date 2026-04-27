use std::{cell::RefCell, cmp::Reverse, rc::Rc};
use gtk4::prelude::*;
use rgis_core::{LayerId, Project};

// ── Sidebar ───────────────────────────────────────────────────────────────────

/// Left-hand panel containing a "Layers" section as a boxed list.
#[derive(Clone)]
pub struct Sidebar {
    root: gtk4::Box,
    list_box: gtk4::ListBox,
    project: Rc<RefCell<Project>>,
    on_change: Rc<dyn Fn()>,
}

impl Sidebar {
    pub fn new(project: Rc<RefCell<Project>>, on_change: impl Fn() + 'static) -> Self {
        let on_change: Rc<dyn Fn()> = Rc::new(on_change);

        let root = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
        root.set_width_request(240);

        // ── Panel header ──────────────────────────────────────────────────────
        let header_label = gtk4::Label::new(Some("Layers"));
        header_label.add_css_class("heading");
        header_label.set_halign(gtk4::Align::Start);
        header_label.set_margin_start(12);
        header_label.set_margin_end(12);
        header_label.set_margin_top(10);
        header_label.set_margin_bottom(10);
        root.append(&header_label);
        root.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        // ── List ──────────────────────────────────────────────────────────────
        let list_box = gtk4::ListBox::new();
        list_box.set_selection_mode(gtk4::SelectionMode::None);

        let scrolled = gtk4::ScrolledWindow::new();
        scrolled.set_vexpand(true);
        scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Automatic);
        scrolled.set_child(Some(&list_box));
        root.append(&scrolled);

        let sidebar = Self { root, list_box, project, on_change };
        sidebar.refresh();
        sidebar
    }

    pub fn widget(&self) -> &gtk4::Box {
        &self.root
    }

    /// Rebuild the list from the current project state.
    pub fn refresh(&self) {
        while let Some(child) = self.list_box.first_child() {
            self.list_box.remove(&child);
        }

        let proj = self.project.borrow();
        let mut layers: Vec<_> = proj.layers.iter().collect();
        layers.sort_by_key(|l| Reverse(l.z_order));

        for layer in &layers {
            let row = self.build_layer_row(layer.id, &layer.name, layer.visible, true);
            self.list_box.append(&row);
        }

        // Background tile row — always last, not removable
        let bg_row = self.build_layer_row(LayerId(u64::MAX), "OSM Background", proj.show_tiles, false);
        self.list_box.append(&bg_row);
    }

    fn build_layer_row(
        &self,
        id: LayerId,
        name: &str,
        visible: bool,
        removable: bool,
    ) -> gtk4::ListBoxRow {
        let row = gtk4::ListBoxRow::new();
        row.set_activatable(false);

        let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        hbox.set_margin_start(6);
        hbox.set_margin_end(6);
        hbox.set_margin_top(6);
        hbox.set_margin_bottom(6);

        // Visibility checkbox
        let check = gtk4::CheckButton::new();
        check.set_active(visible);
        check.set_tooltip_text(Some("Toggle visibility"));
        check.set_valign(gtk4::Align::Center);

        let project = Rc::clone(&self.project);
        let on_change = Rc::clone(&self.on_change);
        check.connect_toggled(move |btn| {
            if id.0 == u64::MAX {
                project.borrow_mut().show_tiles = btn.is_active();
            } else if let Some(layer) = project.borrow_mut().get_layer_mut(id) {
                layer.visible = btn.is_active();
            }
            on_change();
        });

        // Name label
        let label = gtk4::Label::new(Some(name));
        label.set_hexpand(true);
        label.set_xalign(0.0);
        label.set_ellipsize(gtk4::pango::EllipsizeMode::End);

        hbox.append(&check);
        hbox.append(&label);

        // Remove button (only for real layers)
        if removable {
            let remove_btn = gtk4::Button::new();
            remove_btn.set_icon_name("list-remove-symbolic");
            remove_btn.set_tooltip_text(Some("Remove layer"));
            remove_btn.add_css_class("flat");
            remove_btn.add_css_class("circular");
            remove_btn.set_valign(gtk4::Align::Center);

            let project = Rc::clone(&self.project);
            let on_change = Rc::clone(&self.on_change);
            let sidebar = self.clone();
            remove_btn.connect_clicked(move |_| {
                project.borrow_mut().remove_layer(id);
                sidebar.refresh();
                on_change();
            });

            hbox.append(&remove_btn);
        }

        row.set_child(Some(&hbox));
        row
    }
}

