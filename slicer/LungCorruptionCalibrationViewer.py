import json
import logging
import os
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *


# Keep the UI legend and imported segment colors in one place so they cannot
# drift apart.  Diagnostic colors are intentionally distinct from the three
# primary contours (in particular, real_missing is no longer cyan like
# approved).
COLORS = {
    "approved": (0.00, 0.75, 1.00),       # cyan
    "autoseg": (1.00, 0.55, 0.00),        # orange
    "synthetic": (0.25, 0.90, 0.25),      # green
    "real_missing": (0.65, 0.25, 1.00),   # violet
    "real_excess": (1.00, 0.10, 0.20),    # red
    "syn_missing": (1.00, 0.90, 0.10),    # yellow
    "syn_excess": (1.00, 0.20, 0.70),     # magenta
}

PRIMARY = [
    ("approved", "Approved", "approved.nii.gz", COLORS["approved"]),
    ("autoseg", "Actual autoseg", "autoseg.nii.gz", COLORS["autoseg"]),
    ("synthetic", "Synthetic", None, COLORS["synthetic"]),
]
DIAGNOSTICS = [
    ("real_missing", "Real missing", "real_missing.nii.gz", COLORS["real_missing"]),
    ("real_excess", "Real excess", "real_excess.nii.gz", COLORS["real_excess"]),
    ("syn_missing", "Synthetic missing", None, COLORS["syn_missing"]),
    ("syn_excess", "Synthetic excess", None, COLORS["syn_excess"]),
]


class LungCorruptionCalibrationViewer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Lung Corruption Calibration Viewer"
        self.parent.categories = ["Segmentation"]
        self.parent.contributors = ["OpenAI"]
        self.parent.helpText = "Inspect Lung_L real errors and best/neighbor synthetic corruption matches."
        self.parent.acknowledgementText = "AIQA contour-QA calibration workflow."


class LungCorruptionCalibrationViewerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = LungCorruptionCalibrationViewerLogic()
        self.index = {}
        self.items = {}

        box = ctk.ctkCollapsibleButton(); box.text = "Calibration inspection"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.root = ctk.ctkPathLineEdit(); self.root.filters = ctk.ctkPathLineEdit.Dirs
        self.root.settingKey = "LungCorruptionCalibrationViewer/root"
        form.addRow("Inspection root:", self.root)
        self.scan = qt.QPushButton("Scan")
        form.addRow(self.scan)
        self.patient = qt.QComboBox(); form.addRow("Patient:", self.patient)
        self.family = qt.QComboBox(); form.addRow("Family:", self.family)
        self.variant = qt.QComboBox(); form.addRow("Severity:", self.variant)
        self.load = qt.QPushButton("Load")
        form.addRow(self.load)

        vis = ctk.ctkCollapsibleButton(); vis.text = "Visibility"
        self.layout.addWidget(vis)
        vf = qt.QFormLayout(vis)
        self.checks = {}
        for key, label, _, color in PRIMARY + DIAGNOSTICS:
            # Put the legend text on the checkbox itself.  This makes the
            # colored text the actual clickable/toggleable control instead of
            # a detached label next to a blank checkbox.
            cb = qt.QCheckBox(label)
            cb.checked = key in {"approved", "autoseg", "synthetic"}
            r, g, b = [int(round(float(c) * 255.0)) for c in color]
            cb.setStyleSheet(
                f"QCheckBox {{ font-weight: 600; color: rgb({r}, {g}, {b}); }}"
            )
            cb.setToolTip(f"Contour color: RGB({r}, {g}, {b})")
            self.checks[key] = cb
            vf.addRow(cb)

        # Use a plain Qt slider with an integer 0..100 range.  This avoids
        # PythonQt/CTK floating-slider quirks and gives reliable one-percent
        # opacity increments in every supported Slicer build.
        opacityRow = qt.QHBoxLayout()
        self.opacity = qt.QSlider(qt.Qt.Horizontal)
        self.opacity.minimum = 0
        self.opacity.maximum = 100
        self.opacity.singleStep = 1
        self.opacity.pageStep = 5
        self.opacity.value = 30
        self.opacityValueLabel = qt.QLabel("30%")
        self.opacityValueLabel.minimumWidth = 42
        opacityRow.addWidget(self.opacity, 1)
        opacityRow.addWidget(self.opacityValueLabel)
        vf.addRow("2D fill opacity:", opacityRow)

        met = ctk.ctkCollapsibleButton(); met.text = "Metrics"
        self.layout.addWidget(met)
        ml = qt.QVBoxLayout(met)
        self.metrics = qt.QLabel("Load a case."); self.metrics.wordWrap = True; self.metrics.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        ml.addWidget(self.metrics)
        self.status = qt.QLabel(""); self.status.wordWrap=True; self.layout.addWidget(self.status)
        self.layout.addStretch(1)

        self.scan.connect("clicked()", self.onScan)
        self.patient.connect("currentIndexChanged(int)", self.onPatient)
        self.family.connect("currentIndexChanged(int)", self.onFamily)
        self.load.connect("clicked()", self.onLoad)
        self.opacity.connect("valueChanged(int)", self.updateVisibility)
        for cb in self.checks.values():
            cb.connect("toggled(bool)", self.updateVisibility)

    def cleanup(self):
        self.clear()

    def clear(self):
        for n in self.items.values():
            try: slicer.mrmlScene.RemoveNode(n)
            except Exception: pass
        self.items = {}

    def onScan(self):
        self.index = self.logic.scan(Path(self.root.currentPath))
        self.patient.clear()
        for pid in sorted(self.index, key=lambda x: int(x) if str(x).isdigit() else 10**9): self.patient.addItem(str(pid))
        self.onPatient()
        self.status.text = f"Found {len(self.index)} patients."

    def onPatient(self, *_):
        pid = self.patient.currentText
        self.family.clear()
        for fam in sorted(self.index.get(pid, {})): self.family.addItem(fam)
        self.onFamily()

    def onFamily(self, *_):
        pid, fam = self.patient.currentText, self.family.currentText
        self.variant.clear()
        for v in self.index.get(pid, {}).get(fam, []): self.variant.addItem(v)

    def onLoad(self):
        pid, fam, var = self.patient.currentText, self.family.currentText, self.variant.currentText
        if not (pid and fam and var): return
        self.clear()
        try:
            items, metrics = self.logic.load(Path(self.root.currentPath), pid, fam, var)
            self.items = items
            self.logic.applyVisibility(items, self.checks, self.opacityFraction())
            slicer.util.setSliceViewerLayers(background=items["ct"])
            slicer.util.resetSliceViews()
            self.metrics.text = metrics
            self.status.text = f"Loaded patient {pid} / {fam} / {var}."
        except Exception as exc:
            logging.exception("Calibration viewer load failed")
            slicer.util.errorDisplay(str(exc))

    def opacityFraction(self):
        return max(0.0, min(1.0, float(self.opacity.value) / 100.0))

    def updateVisibility(self, *_):
        self.opacityValueLabel.text = f"{int(self.opacity.value)}%"
        if not self.items:
            return
        self.logic.applyVisibility(self.items, self.checks, self.opacityFraction())


class LungCorruptionCalibrationViewerLogic(ScriptedLoadableModuleLogic):
    def scan(self, root: Path):
        out = {}
        if not root.is_dir(): return out
        for pdir in root.glob("patient_*"):
            if not pdir.is_dir(): continue
            pid = pdir.name.replace("patient_", "", 1)
            fams = {}
            for fdir in pdir.iterdir():
                if not fdir.is_dir(): continue
                variants = [v for v in ("best", "lower", "upper") if (fdir / f"{v}.nii.gz").is_file()]
                if variants: fams[fdir.name] = variants
            if fams: out[pid] = fams
        return out

    def _seg(self, ref, name):
        n = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        n.SetName(name); n.CreateDefaultDisplayNodes(); n.SetReferenceImageGeometryParameterFromVolumeNode(ref)
        return n

    def _add(self, path: Path, seg, name: str, color, ref):
        if not path.is_file():
            logging.warning("Calibration viewer: missing labelmap %s", path)
            return None

        label = slicer.util.loadLabelVolume(str(path))
        if label is None:
            raise RuntimeError(f"Could not load labelmap: {path}")

        try:
            segmentation = seg.GetSegmentation()
            before_ids = {
                segmentation.GetNthSegmentID(i)
                for i in range(segmentation.GetNumberOfSegments())
            }

            # Empty missing/excess masks are valid calibration outputs.  Slicer
            # does not create a segment when an all-zero labelmap is imported.
            # Detect that case explicitly instead of indexing past the final
            # segment and dereferencing None.
            try:
                is_empty = not bool(slicer.util.arrayFromVolume(label).any())
            except Exception:
                is_empty = False

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, seg)

            after_ids = [
                segmentation.GetNthSegmentID(i)
                for i in range(segmentation.GetNumberOfSegments())
            ]
            new_ids = [sid for sid in after_ids if sid not in before_ids]

            if not new_ids:
                if is_empty:
                    logging.info(
                        "Calibration viewer: skipping empty labelmap %s (%s)",
                        path,
                        name,
                    )
                    return None
                raise RuntimeError(
                    f"Slicer imported no segment from non-empty labelmap: {path}"
                )

            sid = new_ids[-1]
            segment = segmentation.GetSegment(sid)
            if segment is None:
                raise RuntimeError(
                    f"Imported segment ID {sid!r} is unavailable for labelmap: {path}"
                )
            segment.SetName(name)
            segment.SetColor(*[float(c) for c in color])
            return sid
        finally:
            slicer.mrmlScene.RemoveNode(label)

    def load(self, root: Path, pid: str, family: str, variant: str):
        pdir, fdir = root/f"patient_{pid}", root/f"patient_{pid}"/family
        ct = slicer.util.loadVolume(str(pdir/"ct.nii.gz")); ct.SetName(f"CT-{pid}")
        seg = self._seg(ct, f"LungCalibration-{pid}-{family}-{variant}")
        self._add(pdir/"approved.nii.gz", seg, "approved", COLORS["approved"], ct)
        self._add(pdir/"autoseg.nii.gz", seg, "autoseg", COLORS["autoseg"], ct)
        self._add(fdir/f"{variant}.nii.gz", seg, "synthetic", COLORS["synthetic"], ct)
        diag = self._seg(ct, f"LungCalibrationDiag-{pid}-{family}-{variant}")
        self._add(pdir/"real_missing.nii.gz", diag, "real_missing", COLORS["real_missing"], ct)
        self._add(pdir/"real_excess.nii.gz", diag, "real_excess", COLORS["real_excess"], ct)
        self._add(fdir/f"{variant}_missing.nii.gz", diag, "syn_missing", COLORS["syn_missing"], ct)
        self._add(fdir/f"{variant}_excess.nii.gz", diag, "syn_excess", COLORS["syn_excess"], ct)
        path = fdir/f"{variant}_metrics.json"
        data = json.loads(path.read_text()) if path.is_file() else {}
        def fmt(prefix, d):
            if not d: return f"{prefix}: unavailable"
            return (f"{prefix}\n  1-Dice: {d.get('one_minus_dice', float('nan')):.4f}\n"
                    f"  ASSD: {d.get('assd_mm', float('nan')):.3f} mm\n"
                    f"  HD95: {d.get('hd95_mm', float('nan')):.3f} mm\n"
                    f"  HD99: {d.get('hd99_mm', float('nan')):.3f} mm")
        metrics = fmt("Actual autoseg vs approved", data.get("real")) + "\n\n" + fmt("Synthetic vs approved", data.get("synthetic"))
        return {"ct":ct, "segments":seg, "diagnostics":diag}, metrics

    def applyVisibility(self, items, checks, opacity):
        opacity = max(0.0, min(1.0, float(opacity)))
        for key in ("segments", "diagnostics"):
            n = items.get(key)
            if n is None:
                continue
            d = n.GetDisplayNode()
            if d is None:
                continue

            # Slicer keeps separate master, 2-D fill, 2-D outline, and 3-D
            # visibility flags.  Updating only SetSegmentVisibility() is not
            # sufficient in all Slicer versions, so explicitly synchronize the
            # 2-D flags used by the slice viewers.
            d.SetVisibility(True)
            try:
                d.SetVisibility2D(True)
            except Exception:
                pass
            segs = n.GetSegmentation()
            for i in range(segs.GetNumberOfSegments()):
                sid = segs.GetNthSegmentID(i)
                segment = segs.GetSegment(sid)
                if segment is None:
                    continue
                name = segment.GetName()
                visible = bool(checks[name].checked) if name in checks else True

                d.SetSegmentVisibility(sid, visible)
                try:
                    d.SetSegmentVisibility2DFill(sid, visible)
                    d.SetSegmentVisibility2DOutline(sid, visible)
                    d.SetSegmentOpacity2DFill(sid, opacity)
                    d.SetSegmentOpacity2DOutline(sid, 1.0)
                    d.SetSegmentVisibility3D(sid, False)
                    d.SetSegmentOpacity3D(sid, 0.40)
                except Exception:
                    # Older Slicer builds may not expose every per-view flag,
                    # but the master visibility above still provides a fallback.
                    pass

            d.Modified()
            n.Modified()

        # Force the slice views to repaint immediately when a checkbox or the
        # opacity slider changes rather than waiting for another UI event.
        try:
            slicer.app.processEvents()
        except Exception:
            pass
