use std::{cell::RefCell, rc::Rc};

use gtk4::prelude::*;
use rgis_core::{LayerId, Project};

/// Sidebar containing the layer list.
#[derive(Clone)]
pub struct Sidebar {
    scrolled: gtk4::ScrolledWindow,
    list_box: gtk4::ListBox,
    project: Rc<RefCell<Project>>,
    on_change: Rc<dyn Fn()>,
}

impl Sidebar {
    pub fn new(project: Rc<RefCell<Project>>, on_change: impl Fn() + 'static) -> Self {
        let list_box = gtk4::ListBox::new();
        list_box.set_selection_mode(gtk4::SelectionMode::None);
        list_box.add_css_class("boxed-list");

        let scrolled = gtk4::ScrolledWindow::new();
        scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Automatic);
        scrolled.set_width_request(240);
        scrolled.set_child(Some(&list_box));

        let sidebar = Self {
            scrolled,
            list_box,
            project,
            on_change: Rc::new(on_change),
        };
        sidebar.refresh();
        sidebar
    }

    pub fn widget(&self) -> &gtk4::ScrolledWindow {
        &self.scrolled
    }

    pub fn refresh(&self) {
        // Remove all rows
        while let Some(child) = self.list_box.first_child() {
            self.list_box.remove(&child);
        }

        let proj = self.project.borrow();
        let mut layers: Vec<_> = proj.layers.iter().collect();
        // Show highest z_order on top
        layers.sort_by_key(|l| std::cmp::Reverse(l.z_order));

        for layer in layers {
            let row = self.build_row(layer.id, &layer.name, layer.visible);
            self.list_box.append(&row);
        }
    }

    fn build_row(&self, id: LayerId, name: &str, visible: bool) -> gtk4::ListBoxRow {
        let row = gtk4::ListBoxRow::new();
        row.set_activatable(false);

        let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        hbox.set_margin_start(8);
        hbox.set_margin_end(8);
        hbox.set_margin_top(6);
        hbox.set_margin_bottom(6);

        // Visibility toggle
        let vis_btn = gtk4::CheckButton::new();
        vis_btn.set_active(visible);
        vis_btn.set_tooltip_text(Some("Toggle visibility"));

        let project = Rc::clone(&self.project);
        let on_change = Rc::clone(&self.on_change);
        vis_btn.connect_toggled(move |btn| {
            if let Some(layer) = project.borrow_mut().get_layer_mut(id) {
                layer.visible = btn.is_active();
            }
            on_change();
        });

        // Name label
        let label = gtk4::Label::new(Some(name));
        label.set_hexpand(true);
        label.set_xalign(0.0);
        label.set_ellipsize(gtk4::pango::EllipsizeMode::End);

        // Remove button
        let remove_btn = gtk4::Button::new();
        remove_btn.set_icon_name("list-remove-symbolic");
        remove_btn.set_tooltip_text(Some("Remove layer"));
        remove_btn.add_css_class("flat");
        remove_btn.add_css_class("circular");

        let project = Rc::clone(&self.project);
        let on_change = Rc::clone(&self.on_change);
        let sidebar = self.clone();
        remove_btn.connect_clicked(move |_| {
            project.borrow_mut().remove_layer(id);
            sidebar.refresh();
            on_change();
        });

        hbox.append(&vis_btn);
        hbox.append(&label);
        hbox.append(&remove_btn);
        row.set_child(Some(&hbox));
        row
    }
}
