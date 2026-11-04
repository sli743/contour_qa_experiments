import json
import logging
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *


COLORS = {
    "approved": (0.00, 0.75, 1.00),       # cyan
    "autoseg": (1.00, 0.55, 0.00),        # orange
    "real_missing": (0.65, 0.25, 1.00),   # violet
    "real_excess": (1.00, 0.10, 0.20),    # red
    "surface_inward": (0.20, 0.45, 1.00), # blue
    "surface_outward": (0.15, 0.85, 0.35),# green
    "smooth_inward": (0.20, 0.75, 1.00),  # sky blue
    "smooth_outward": (0.15, 0.65, 0.15), # dark green
    "local_inward": (1.00, 0.85, 0.10),   # yellow
    "local_outward": (1.00, 0.20, 0.70),  # magenta
}

MASKS = [
    ("approved", "Approved", "approved.nii.gz", COLORS["approved"], True),
    ("autoseg", "Actual autoseg", "autoseg.nii.gz", COLORS["autoseg"], True),
    ("real_missing", "Real missing", "real_missing.nii.gz", COLORS["real_missing"], False),
    ("real_excess", "Real excess", "real_excess.nii.gz", COLORS["real_excess"], False),
    ("surface_inward", "Surface inward ≥ threshold", "surface_inward_ge_threshold.nii.gz", COLORS["surface_inward"], False),
    ("surface_outward", "Surface outward ≥ threshold", "surface_outward_ge_threshold.nii.gz", COLORS["surface_outward"], False),
    ("smooth_inward", "Smooth inward ≥ threshold", "smooth_inward_ge_threshold.nii.gz", COLORS["smooth_inward"], False),
    ("smooth_outward", "Smooth outward ≥ threshold", "smooth_outward_ge_threshold.nii.gz", COLORS["smooth_outward"], False),
    ("local_inward", "Local inward ≥ threshold", "local_inward_ge_threshold.nii.gz", COLORS["local_inward"], False),
    ("local_outward", "Local outward ≥ threshold", "local_outward_ge_threshold.nii.gz", COLORS["local_outward"], False),
]


class LungErrorFieldViewer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Lung Error Field Viewer"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["OpenAI"]
        self.parent.helpText = "Inspect shortlisted Lung_L real autoseg error-field cases."
        self.parent.acknowledgementText = "AIQA contour-QA workflow."


class LungErrorFieldViewerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = LungErrorFieldViewerLogic()
        self.index = {}
        self.items = {}

        box = ctk.ctkCollapsibleButton(); box.text = "Error-field inspection"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.root = ctk.ctkPathLineEdit(); self.root.filters = ctk.ctkPathLineEdit.Dirs
        self.root.settingKey = "LungErrorFieldViewer/root"
        form.addRow("Inspection root:", self.root)
        self.scan = qt.QPushButton("Scan")
        form.addRow(self.scan)
        self.patient = qt.QComboBox(); form.addRow("Patient:", self.patient)
        self.load = qt.QPushButton("Load")
        form.addRow(self.load)

        vis = ctk.ctkCollapsibleButton(); vis.text = "Visibility"
        self.layout.addWidget(vis)
        vf = qt.QFormLayout(vis)
        self.checks = {}
        for key, label, _filename, color, default in MASKS:
            cb = qt.QCheckBox(label)
            cb.checked = bool(default)
            r, g, b = [int(round(float(c) * 255.0)) for c in color]
            cb.setStyleSheet(f"QCheckBox {{ font-weight: 600; color: rgb({r}, {g}, {b}); }}")
            cb.setToolTip(f"Contour color: RGB({r}, {g}, {b})")
            self.checks[key] = cb
            vf.addRow(cb)

        opacityRow = qt.QHBoxLayout()
        self.opacity = qt.QSlider(qt.Qt.Horizontal)
        self.opacity.minimum = 0; self.opacity.maximum = 100
        self.opacity.singleStep = 1; self.opacity.pageStep = 5; self.opacity.value = 25
        self.opacityValueLabel = qt.QLabel("25%"); self.opacityValueLabel.minimumWidth = 42
        opacityRow.addWidget(self.opacity, 1); opacityRow.addWidget(self.opacityValueLabel)
        vf.addRow("2D fill opacity:", opacityRow)

        met = ctk.ctkCollapsibleButton(); met.text = "Case summary"
        self.layout.addWidget(met)
        ml = qt.QVBoxLayout(met)
        self.metrics = qt.QLabel("Load a case.")
        self.metrics.wordWrap = True
        self.metrics.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        ml.addWidget(self.metrics)
        self.status = qt.QLabel(""); self.status.wordWrap = True; self.layout.addWidget(self.status)
        self.layout.addStretch(1)

        self.scan.connect("clicked()", self.onScan)
        self.load.connect("clicked()", self.onLoad)
        self.opacity.connect("valueChanged(int)", self.updateVisibility)
        for cb in self.checks.values(): cb.connect("toggled(bool)", self.updateVisibility)

    def cleanup(self):
        self.clear()

    def clear(self):
        for n in self.items.values():
            try: slicer.mrmlScene.RemoveNode(n)
            except Exception: pass
        self.items = {}

    def opacityFraction(self):
        return max(0.0, min(1.0, float(self.opacity.value) / 100.0))

    def onScan(self):
        self.index = self.logic.scan(Path(self.root.currentPath))
        self.patient.clear()
        for pid in sorted(self.index, key=lambda x: int(x) if str(x).isdigit() else 10**9):
            self.patient.addItem(str(pid))
        self.status.text = f"Found {len(self.index)} exported patients."

    def onLoad(self):
        pid = self.patient.currentText
        if not pid: return
        self.clear()
        try:
            items, text = self.logic.load(Path(self.root.currentPath), pid)
            self.items = items
            self.logic.applyVisibility(items, self.checks, self.opacityFraction())
            slicer.util.setSliceViewerLayers(background=items["ct"])
            slicer.util.resetSliceViews()
            self.metrics.text = text
            self.status.text = f"Loaded patient {pid}."
        except Exception as exc:
            logging.exception("Lung error-field viewer load failed")
            slicer.util.errorDisplay(str(exc))

    def updateVisibility(self, *_):
        self.opacityValueLabel.text = f"{int(self.opacity.value)}%"
        if self.items:
            self.logic.applyVisibility(self.items, self.checks, self.opacityFraction())


class LungErrorFieldViewerLogic(ScriptedLoadableModuleLogic):
    def scan(self, root: Path):
        out = {}
        if not root.is_dir(): return out
        for p in root.glob("patient_*"):
            if p.is_dir() and (p / "approved.nii.gz").is_file() and (p / "autoseg.nii.gz").is_file():
                out[p.name.replace("patient_", "", 1)] = p
        return out

    def _seg(self, ref, name):
        n = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        n.SetName(name); n.CreateDefaultDisplayNodes(); n.SetReferenceImageGeometryParameterFromVolumeNode(ref)
        return n

    def _add(self, path: Path, seg, name: str, color):
        if not path.is_file():
            logging.warning("Lung error-field viewer: missing %s", path)
            return None
        label = slicer.util.loadLabelVolume(str(path))
        if label is None:
            raise RuntimeError(f"Could not load labelmap: {path}")
        try:
            segmentation = seg.GetSegmentation()
            before = {segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())}
            try: is_empty = not bool(slicer.util.arrayFromVolume(label).any())
            except Exception: is_empty = False
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, seg)
            after = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]
            new_ids = [sid for sid in after if sid not in before]
            if not new_ids:
                if is_empty:
                    return None
                raise RuntimeError(f"Slicer imported no segment from non-empty labelmap: {path}")
            sid = new_ids[-1]
            segment = segmentation.GetSegment(sid)
            if segment is None:
                raise RuntimeError(f"Imported segment ID {sid!r} unavailable for {path}")
            segment.SetName(name); segment.SetColor(*[float(c) for c in color])
            return sid
        finally:
            slicer.mrmlScene.RemoveNode(label)

    def load(self, root: Path, pid: str):
        pdir = root / f"patient_{pid}"
        ct = slicer.util.loadVolume(str(pdir / "ct.nii.gz"))
        if ct is None: raise RuntimeError(f"Could not load CT for patient {pid}")
        ct.SetName(f"CT-{pid}")
        seg = self._seg(ct, f"LungErrorField-{pid}")
        ids = {}
        for key, label, filename, color, _default in MASKS:
            sid = self._add(pdir / filename, seg, key, color)
            if sid: ids[key] = sid

        path = pdir / "metrics.json"
        data = json.loads(path.read_text()) if path.is_file() else {}
        s = data.get("summary", {})
        threshold = data.get("surface_display_threshold_mm", "NA")
        text = (
            f"Patient {pid} | domain: {data.get('domain', 'NA')}\n"
            f"Inspection categories: {data.get('inspection_categories', '')}\n"
            f"Why inspect: {data.get('inspection_reasons', '')}\n\n"
            f"1-Dice: {s.get('one_minus_dice', float('nan')):.4f}\n"
            f"ASSD: {s.get('assd_mm', float('nan')):.3f} mm\n"
            f"HD99: {s.get('hd99_mm', float('nan')):.3f} mm\n"
            f"Missing / excess fraction: {s.get('missing_fraction_gt', float('nan')):.4f} / {s.get('excess_fraction_gt', float('nan')):.4f}\n"
            f"Surface displacement RMS: {s.get('surface_displacement_rms_mm', float('nan')):.3f} mm\n"
            f"Smooth / local RMS: {s.get('smooth_rms_mm', float('nan')):.3f} / {s.get('local_rms_mm', float('nan')):.3f} mm\n"
            f"Max S/I endpoint error: {s.get('max_abs_endpoint_error_mm', float('nan')):.3f} mm\n"
            f"Significant patches: {s.get('total_component_count_significant', 0)}\n"
            f"Displayed inward/outward threshold: {threshold} mm"
        )
        return {"ct": ct, "segments": seg, "ids": ids}, text

    def applyVisibility(self, items, checks, opacity):
        seg = items.get("segments")
        if seg is None: return
        display = seg.GetDisplayNode()
        if display is None: return
        ids = items.get("ids", {})
        opacity = max(0.0, min(1.0, float(opacity)))
        display.SetVisibility(True); display.SetVisibility2D(True)
        for key, sid in ids.items():
            visible = bool(checks[key].checked)
            display.SetSegmentVisibility(sid, visible)
            try:
                display.SetSegmentVisibility2DFill(sid, visible)
                display.SetSegmentVisibility2DOutline(sid, visible)
                display.SetSegmentOpacity2DFill(sid, opacity)
                display.SetSegmentOpacity2DOutline(sid, 1.0)
                display.SetSegmentOpacity3D(sid, 0.45)
            except Exception:
                pass
        try: display.Modified()
        except Exception: pass
