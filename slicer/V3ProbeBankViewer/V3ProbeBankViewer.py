import csv
import json
import logging
import os
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *


DEFAULT_ROOT = r"D:\aiqa\contour_QA\spinalcord_v3_probe_examples"

CONTOUR_SPECS = [
    ("approved", "Approved", "approved.nii.gz", (0.00, 0.85, 1.00)),
    ("autoseg", "Actual autoseg", "autoseg.nii.gz", (1.00, 0.55, 0.00)),
    ("probe", "V3 synthetic probe", None, (0.95, 0.10, 0.70)),
]


class V3ProbeBankViewer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SpinalCord V3 Probe Bank Viewer"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["OpenAI"]
        self.parent.helpText = """
Inspect exact SpinalCord v3 synthetic probe-bank examples beside the corresponding
approved contour and actual clinical autosegmentation.  Cases are exported by
analysis/export_spinalcord_v3_probe_examples.py; this Slicer module only reads
those NIfTI/CSV/JSON files and does not regenerate corruptions.
"""
        self.parent.acknowledgementText = "Generated for the AIQA contour-QA workflow."


class V3ProbeBankViewerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = V3ProbeBankViewerLogic()
        self.caseIndex = {}
        self.loadedNodes = {"ct": None, "seg": None}
        self.segmentIds = {}
        self.currentCase = None
        self.currentProbeRows = []

        inputBox = ctk.ctkCollapsibleButton()
        inputBox.text = "V3 probe examples"
        self.layout.addWidget(inputBox)
        form = qt.QFormLayout(inputBox)

        self.directoryButton = ctk.ctkPathLineEdit()
        self.directoryButton.filters = ctk.ctkPathLineEdit.Dirs
        self.directoryButton.settingKey = "V3ProbeBankViewer/rootDir"
        self.directoryButton.currentPath = DEFAULT_ROOT
        form.addRow("Export root:", self.directoryButton)

        self.scanButton = qt.QPushButton("Scan directory")
        form.addRow(self.scanButton)

        self.patientCombo = qt.QComboBox()
        form.addRow("Bad-patient rank:", self.patientCombo)

        patientRow = qt.QHBoxLayout()
        self.prevPatientButton = qt.QPushButton("Previous patient")
        self.nextPatientButton = qt.QPushButton("Next patient")
        patientRow.addWidget(self.prevPatientButton)
        patientRow.addWidget(self.nextPatientButton)
        form.addRow(patientRow)

        self.probeCombo = qt.QComboBox()
        form.addRow("Closest same-patient probe:", self.probeCombo)

        probeRow = qt.QHBoxLayout()
        self.prevProbeButton = qt.QPushButton("Previous probe")
        self.nextProbeButton = qt.QPushButton("Next probe")
        probeRow.addWidget(self.prevProbeButton)
        probeRow.addWidget(self.nextProbeButton)
        form.addRow(probeRow)

        buttonRow = qt.QHBoxLayout()
        self.loadButton = qt.QPushButton("Load patient + selected probe")
        self.clearButton = qt.QPushButton("Clear loaded nodes")
        buttonRow.addWidget(self.loadButton)
        buttonRow.addWidget(self.clearButton)
        form.addRow(buttonRow)

        visBox = ctk.ctkCollapsibleButton()
        visBox.text = "Visibility"
        self.layout.addWidget(visBox)
        visForm = qt.QFormLayout(visBox)

        self.showCT = qt.QCheckBox()
        self.showCT.checked = True
        visForm.addRow("Show CT:", self.showCT)

        self.contourChecks = {}
        for key, label, _filename, color in CONTOUR_SPECS:
            cb = qt.QCheckBox()
            cb.checked = True
            self.contourChecks[key] = cb
            visForm.addRow(self._colorLabel(label, color), cb)

        self.opacity = ctk.ctkSliderWidget()
        self.opacity.minimum = 0.0
        self.opacity.maximum = 1.0
        self.opacity.singleStep = 0.05
        self.opacity.value = 0.20
        self.opacity.decimals = 2
        visForm.addRow("2D fill opacity:", self.opacity)

        self.thickness = ctk.ctkSliderWidget()
        self.thickness.minimum = 1
        self.thickness.maximum = 10
        self.thickness.singleStep = 1
        self.thickness.value = 2
        self.thickness.decimals = 0
        visForm.addRow("2D contour thickness:", self.thickness)

        self.show3D = qt.QCheckBox()
        self.show3D.checked = False
        visForm.addRow("Show contours in 3D:", self.show3D)

        self.fitButton = qt.QPushButton("Fit slice views")
        visForm.addRow(self.fitButton)

        metricsBox = ctk.ctkCollapsibleButton()
        metricsBox.text = "Real error and selected probe"
        self.layout.addWidget(metricsBox)
        metricsLayout = qt.QVBoxLayout(metricsBox)
        self.metricsLabel = qt.QLabel("Scan the export root and choose a case.")
        self.metricsLabel.wordWrap = True
        self.metricsLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        metricsLayout.addWidget(self.metricsLabel)

        self.statusLabel = qt.QLabel("Select the v3 probe-example export root and scan.")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)

        self.scanButton.connect("clicked()", self.onScan)
        self.patientCombo.connect("currentIndexChanged(int)", self.onPatientChanged)
        self.probeCombo.connect("currentIndexChanged(int)", self.onProbeChanged)
        self.prevPatientButton.connect("clicked()", lambda: self._stepCombo(self.patientCombo, -1))
        self.nextPatientButton.connect("clicked()", lambda: self._stepCombo(self.patientCombo, 1))
        self.prevProbeButton.connect("clicked()", lambda: self._stepCombo(self.probeCombo, -1))
        self.nextProbeButton.connect("clicked()", lambda: self._stepCombo(self.probeCombo, 1))
        self.loadButton.connect("clicked()", self.onLoad)
        self.clearButton.connect("clicked()", self.clearLoadedNodes)
        self.fitButton.connect("clicked()", self.fitViews)
        self.showCT.connect("toggled(bool)", self.updateVisibility)
        self.opacity.connect("valueChanged(double)", self.updateVisibility)
        self.thickness.connect("valueChanged(double)", self.updateVisibility)
        self.show3D.connect("toggled(bool)", self.updateVisibility)
        for cb in self.contourChecks.values():
            cb.connect("toggled(bool)", self.updateVisibility)

        if os.path.isdir(DEFAULT_ROOT):
            self.onScan()

    @staticmethod
    def _colorLabel(text, color):
        r, g, b = [int(round(c * 255)) for c in color]
        label = qt.QLabel(text)
        label.setStyleSheet(f"QLabel {{ font-weight: 600; color: rgb({r},{g},{b}); }}")
        return label

    @staticmethod
    def _stepCombo(combo, delta):
        if combo.count <= 0:
            return
        combo.setCurrentIndex(max(0, min(combo.count - 1, combo.currentIndex + delta)))

    def cleanup(self):
        self.clearLoadedNodes()

    def onScan(self):
        root = Path(self.directoryButton.currentPath)
        self.caseIndex = self.logic.scanRoot(root)
        self.patientCombo.blockSignals(True)
        self.patientCombo.clear()
        for case in self.caseIndex:
            pid = case["patient_id"]
            rank = case.get("patient_rank", "?")
            hd95 = self.logic.formatNumber(case.get("real_hd95_mm"), 2)
            assd = self.logic.formatNumber(case.get("real_assd_mm"), 2)
            inf = self.logic.formatNumber(case.get("real_inferior_missing_mm"), 1)
            self.patientCombo.addItem(
                f"#{rank} patient {pid} | HD95 {hd95} mm | ASSD {assd} | inferior miss {inf} mm",
                str(pid),
            )
        self.patientCombo.blockSignals(False)
        self.onPatientChanged()
        self.statusLabel.text = f"Found {len(self.caseIndex)} exported bad-patient case(s)."

    def onPatientChanged(self, *_args):
        self.currentCase = None
        self.currentProbeRows = []
        self.probeCombo.blockSignals(True)
        self.probeCombo.clear()
        idx = self.patientCombo.currentIndex
        if 0 <= idx < len(self.caseIndex):
            self.currentCase = self.caseIndex[idx]
            caseDir = Path(self.currentCase["case_dir"])
            self.currentProbeRows = self.logic.readProbeManifest(caseDir)
            for row in self.currentProbeRows:
                rank = row.get("probe_rank", "?")
                sample = row.get("sample_index", "?")
                dist = self.logic.formatNumber(row.get("phenotype_distance"), 3)
                composition = row.get("composition_label", row.get("branch", ""))
                hd95 = self.logic.formatNumber(row.get("probe_hd95_mm"), 2)
                inf = self.logic.formatNumber(row.get("probe_inferior_missing_mm"), 1)
                self.probeCombo.addItem(
                    f"#{rank} sample {sample} | d={dist} | {composition} | HD95 {hd95} | inf miss {inf}",
                    row.get("filename", ""),
                )
        self.probeCombo.blockSignals(False)
        self.onProbeChanged()
        self.loadButton.enabled = bool(self.currentCase and self.currentProbeRows)

    def onProbeChanged(self, *_args):
        if not self.currentCase or not self.currentProbeRows:
            self.metricsLabel.text = "No probe selected."
            return
        idx = self.probeCombo.currentIndex
        idx = max(0, min(idx, len(self.currentProbeRows) - 1))
        info = self.logic.readCaseInfo(Path(self.currentCase["case_dir"]))
        self.metricsLabel.text = self.logic.formatMetrics(info, self.currentProbeRows[idx])

    def onLoad(self):
        if not self.currentCase or not self.currentProbeRows:
            slicer.util.errorDisplay("No exported v3 probe case is selected.")
            return
        probe_idx = self.probeCombo.currentIndex
        probe_idx = max(0, min(probe_idx, len(self.currentProbeRows) - 1))
        probeRow = self.currentProbeRows[probe_idx]
        caseDir = Path(self.currentCase["case_dir"])
        self.clearLoadedNodes()
        try:
            ct, seg, ids = self.logic.loadCase(caseDir, probeRow)
            self.loadedNodes = {"ct": ct, "seg": seg}
            self.segmentIds = ids
            slicer.util.setSliceViewerLayers(background=ct)
            self.updateVisibility()
            self.fitViews()
            info = self.logic.readCaseInfo(caseDir)
            self.metricsLabel.text = self.logic.formatMetrics(info, probeRow)
            self.statusLabel.text = (
                f"Loaded patient {self.currentCase['patient_id']}, "
                f"v3 sample {probeRow.get('sample_index', '?')} "
                f"(same-patient phenotype distance {self.logic.formatNumber(probeRow.get('phenotype_distance'), 3)})."
            )
        except Exception as exc:
            logging.exception("Failed to load v3 probe example")
            slicer.util.errorDisplay(str(exc))

    def updateVisibility(self, *_args):
        ct = self.loadedNodes.get("ct")
        if ct and ct.GetDisplayNode():
            ct.GetDisplayNode().SetVisibility(bool(self.showCT.checked))
        seg = self.loadedNodes.get("seg")
        if not seg or not seg.GetDisplayNode():
            return
        display = seg.GetDisplayNode()
        display.SetVisibility(True)
        display.SetVisibility2D(True)
        display.SetVisibility3D(bool(self.show3D.checked))
        try:
            display.SetSliceIntersectionThickness(max(1, int(round(float(self.thickness.value)))))
        except Exception:
            pass
        for key, segId in self.segmentIds.items():
            display.SetSegmentVisibility(segId, bool(self.contourChecks[key].checked))
            try:
                display.SetSegmentOpacity2DFill(segId, float(self.opacity.value))
                display.SetSegmentOpacity2DOutline(segId, 1.0)
                display.SetSegmentOpacity3D(segId, 0.55)
            except Exception:
                pass

    def fitViews(self):
        try:
            slicer.util.resetSliceViews()
        except Exception:
            pass

    def clearLoadedNodes(self):
        for key in ("seg", "ct"):
            node = self.loadedNodes.get(key)
            if node:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
        self.loadedNodes = {"ct": None, "seg": None}
        self.segmentIds = {}


class V3ProbeBankViewerLogic(ScriptedLoadableModuleLogic):
    @staticmethod
    def _float(value):
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def formatNumber(value, digits=3):
        try:
            value = float(value)
            if value != value:
                return "NA"
            return f"{value:.{digits}f}"
        except Exception:
            return "NA"

    def scanRoot(self, root: Path):
        if not root.is_dir():
            return []
        manifest = root / "manifest.csv"
        rows = []
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        pid = int(float(row["patient_id"]))
                    except Exception:
                        continue
                    caseDir = root / f"patient_{pid}"
                    if not self._validCase(caseDir):
                        continue
                    row = dict(row)
                    row["patient_id"] = pid
                    row["case_dir"] = str(caseDir)
                    rows.append(row)
            rows.sort(key=lambda r: int(float(r.get("patient_rank", 999999))))
            return rows

        for caseDir in sorted(root.glob("patient_*")):
            if not self._validCase(caseDir):
                continue
            try:
                pid = int(caseDir.name.replace("patient_", "", 1))
            except Exception:
                continue
            info = self.readCaseInfo(caseDir)
            full = info.get("real_full_volume_features", {})
            rows.append({
                "patient_id": pid,
                "patient_rank": info.get("patient_rank", len(rows) + 1),
                "real_hd95_mm": full.get("hd95_mm"),
                "real_assd_mm": full.get("assd_mm"),
                "real_inferior_missing_mm": full.get("inferior_missing_mm"),
                "case_dir": str(caseDir),
            })
        rows.sort(key=lambda r: int(r.get("patient_rank", 999999)))
        return rows

    @staticmethod
    def _validCase(caseDir: Path):
        return (
            caseDir.is_dir()
            and (caseDir / "ct.nii.gz").is_file()
            and (caseDir / "approved.nii.gz").is_file()
            and (caseDir / "autoseg.nii.gz").is_file()
            and (caseDir / "probe_manifest.csv").is_file()
        )

    @staticmethod
    def readProbeManifest(caseDir: Path):
        path = caseDir / "probe_manifest.csv"
        if not path.is_file():
            return []
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("filename") and (caseDir / row["filename"]).is_file():
                    rows.append(dict(row))
        rows.sort(key=lambda r: int(float(r.get("probe_rank", 999999))))
        return rows

    @staticmethod
    def readCaseInfo(caseDir: Path):
        path = caseDir / "case_info.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _loadLabel(self, path: Path, node, name, color, reference):
        if not path.is_file():
            raise FileNotFoundError(path)
        label = slicer.util.loadLabelVolume(str(path))
        if not label:
            raise RuntimeError(f"Could not load labelmap: {path}")
        try:
            seg = node.GetSegmentation()
            before = {seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())}
            node.SetReferenceImageGeometryParameterFromVolumeNode(reference)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, node)
            after = [seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())]
            newIds = [sid for sid in after if sid not in before]
            if not newIds:
                raise RuntimeError(f"No segment imported from: {path}")
            sid = newIds[-1]
            segment = seg.GetSegment(sid)
            segment.SetName(name)
            segment.SetColor(*[float(c) for c in color])
            return sid
        finally:
            slicer.mrmlScene.RemoveNode(label)

    def loadCase(self, caseDir: Path, probeRow):
        ctPath = caseDir / "ct.nii.gz"
        ct = slicer.util.loadVolume(str(ctPath))
        if not ct:
            raise RuntimeError(f"Could not load CT: {ctPath}")
        pid = caseDir.name.replace("patient_", "", 1)
        ct.SetName(f"V3Probe-CT-patient_{pid}")

        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg.SetName(f"V3ProbeComparison-patient_{pid}")
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        ids = {}
        for key, name, filename, color in CONTOUR_SPECS:
            path = caseDir / (probeRow["filename"] if key == "probe" else filename)
            ids[key] = self._loadLabel(path, seg, name, color, ct)
        try:
            seg.CreateClosedSurfaceRepresentation()
        except Exception:
            pass
        return ct, seg, ids

    def formatMetrics(self, info, probeRow):
        full = info.get("real_full_volume_features", {})
        view = info.get("real_prepared_grid_features", {})

        def f(mapping, key, digits=3):
            return self.formatNumber(mapping.get(key), digits)

        lines = [
            "REAL AUTOSEG ERROR — full-volume calibration phenotype",
            f"  Dice: {f(full, 'dice', 4)}   1-Dice: {self._oneMinusDice(full.get('dice'))}",
            f"  HD95: {f(full, 'hd95_mm')} mm   ASSD: {f(full, 'assd_mm')} mm",
            f"  Missing / excess: {f(full, 'missing_fraction_gt', 4)} / {f(full, 'excess_fraction_gt', 4)}",
            f"  Inferior miss/excess: {f(full, 'inferior_missing_mm', 1)} / {f(full, 'inferior_excess_mm', 1)} mm",
            f"  Superior miss/excess: {f(full, 'superior_missing_mm', 1)} / {f(full, 'superior_excess_mm', 1)} mm",
            "",
            "REAL AUTOSEG ERROR — prepared-grid view shown in Slicer",
            f"  Dice: {f(view, 'dice', 4)}   HD95: {f(view, 'hd95_mm')} mm   ASSD: {f(view, 'assd_mm')} mm",
            f"  Inferior missing: {f(view, 'inferior_missing_mm', 1)} mm",
            "",
            "SELECTED V3 PROBE",
            f"  Same-patient phenotype distance: {self.formatNumber(probeRow.get('phenotype_distance'), 4)}",
            f"  Bank sample: {probeRow.get('sample_index', 'NA')}   branch: {probeRow.get('branch', '')}",
            f"  Composition: {probeRow.get('composition_label', '')}",
            f"  Surface: {probeRow.get('surface_mode', '')}   endpoint: {probeRow.get('endpoint_event', '')}   local bite: {probeRow.get('local_bite_applied', '')}",
            f"  Dice: {self.formatNumber(probeRow.get('probe_dice'), 4)}   HD95: {self.formatNumber(probeRow.get('probe_hd95_mm'))} mm   ASSD: {self.formatNumber(probeRow.get('probe_assd_mm'))} mm",
            f"  Missing / excess: {self.formatNumber(probeRow.get('probe_missing_fraction_gt'), 4)} / {self.formatNumber(probeRow.get('probe_excess_fraction_gt'), 4)}",
            f"  Inferior missing: {self.formatNumber(probeRow.get('probe_inferior_missing_mm'), 1)} mm",
            "",
            "DISTANCE BY PHENOTYPE GROUP",
        ]
        distance_items = sorted(
            (k.replace("distance_", ""), v)
            for k, v in probeRow.items()
            if k.startswith("distance_") and k != "distance_method"
        )
        if distance_items:
            for name, value in distance_items:
                lines.append(f"  {name}: {self.formatNumber(value, 4)}")
        else:
            lines.append("  NA")
        return "\n".join(lines)

    def _oneMinusDice(self, value):
        try:
            return f"{1.0 - float(value):.4f}"
        except Exception:
            return "NA"


class V3ProbeBankViewerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.delayDisplay("SpinalCord V3 Probe Bank Viewer loaded")
