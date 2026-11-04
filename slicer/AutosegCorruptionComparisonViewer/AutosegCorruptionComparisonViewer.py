import logging
import os
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *


DEFAULT_NO_CORRUPTION_ROOT = (
    r"D:\aiqa\contour_QA\lung_l_nsrqa_corruption_ablation_v1\evaluations"
    r"\no_corruption\seed91\actual_autoseg_cases"
)
DEFAULT_NSRQA_ROOT = (
    r"D:\aiqa\contour_QA\lung_l_nsrqa_corruption_ablation_v1\evaluations"
    r"\nsrqa_corruption\seed91\actual_autoseg_cases"
)

SEGMENT_SPECS = [
    ("approved", "Approved", (0.00, 0.85, 1.00)),
    ("autoseg", "Autoseg input", (1.00, 0.55, 0.00)),
    ("baseline_recon", "No-corruption reconstruction", (0.75, 0.35, 1.00)),
    ("nsrqa_recon", "NSR-QA-corruption reconstruction", (0.30, 0.85, 0.25)),
]


class AutosegCorruptionComparisonViewer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Autoseg Corruption Comparison Viewer"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["OpenAI"]
        self.parent.helpText = """
Compare actual-autoseg reconstruction outputs from a no-corruption shape model
and an NSR-QA-corruption shape model. The module pairs cases by localization
mode, ROI, split, and patient ID and displays only four contours: approved,
autoseg input, no-corruption reconstruction, and NSR-QA-corruption
reconstruction.

Each selected root must be an ``actual_autoseg_cases`` directory with layout:
  <root>/<mode>/<roi>/<split>/patient_<id>/
"""
        self.parent.acknowledgementText = "Generated for the AIQA contour-QA workflow."


class AutosegCorruptionComparisonViewerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = AutosegCorruptionComparisonViewerLogic()
        self.caseIndex = {}
        self.loadedNodes = {"ct": None, "seg": None}
        self.segmentIds = {}

        inputBox = ctk.ctkCollapsibleButton()
        inputBox.text = "Paired experiment outputs"
        self.layout.addWidget(inputBox)
        form = qt.QFormLayout(inputBox)

        self.baselineDirectoryButton = ctk.ctkPathLineEdit()
        self.baselineDirectoryButton.filters = ctk.ctkPathLineEdit.Dirs
        self.baselineDirectoryButton.settingKey = "AutosegCorruptionComparisonViewer/noCorruptionRoot"
        self.baselineDirectoryButton.currentPath = DEFAULT_NO_CORRUPTION_ROOT
        form.addRow("No-corruption cases root:", self.baselineDirectoryButton)

        self.nsrqaDirectoryButton = ctk.ctkPathLineEdit()
        self.nsrqaDirectoryButton.filters = ctk.ctkPathLineEdit.Dirs
        self.nsrqaDirectoryButton.settingKey = "AutosegCorruptionComparisonViewer/nsrqaRoot"
        self.nsrqaDirectoryButton.currentPath = DEFAULT_NSRQA_ROOT
        form.addRow("NSR-QA cases root:", self.nsrqaDirectoryButton)

        self.scanButton = qt.QPushButton("Scan paired directories")
        form.addRow(self.scanButton)

        self.modeCombo = qt.QComboBox()
        form.addRow("Localization mode:", self.modeCombo)
        self.roiCombo = qt.QComboBox()
        form.addRow("ROI:", self.roiCombo)
        self.splitCombo = qt.QComboBox()
        form.addRow("Split:", self.splitCombo)
        self.patientCombo = qt.QComboBox()
        form.addRow("Patient:", self.patientCombo)

        navigationRow = qt.QHBoxLayout()
        self.previousButton = qt.QPushButton("Previous")
        self.loadButton = qt.QPushButton("Load selected case")
        self.nextButton = qt.QPushButton("Next")
        navigationRow.addWidget(self.previousButton)
        navigationRow.addWidget(self.loadButton)
        navigationRow.addWidget(self.nextButton)
        form.addRow(navigationRow)

        self.clearButton = qt.QPushButton("Clear loaded nodes")
        form.addRow(self.clearButton)

        visBox = ctk.ctkCollapsibleButton()
        visBox.text = "Visibility"
        self.layout.addWidget(visBox)
        visForm = qt.QFormLayout(visBox)

        self.showCT = qt.QCheckBox()
        self.showCT.checked = True
        visForm.addRow("Show CT:", self.showCT)

        self.segmentChecks = {}
        for key, label, color in SEGMENT_SPECS:
            cb = qt.QCheckBox()
            cb.checked = True
            self.segmentChecks[key] = cb
            visForm.addRow(self._colorLabel(label, color), cb)

        self.opacity = ctk.ctkSliderWidget()
        self.opacity.minimum = 0.0
        self.opacity.maximum = 1.0
        self.opacity.singleStep = 0.05
        self.opacity.value = 0.25
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

        self.statusLabel = qt.QLabel(
            "Select the no-corruption and NSR-QA actual_autoseg_cases roots, then scan."
        )
        self.statusLabel.wordWrap = True
        self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)

        self.scanButton.connect("clicked()", self.onScan)
        self.modeCombo.connect("currentIndexChanged(int)", self.onModeChanged)
        self.roiCombo.connect("currentIndexChanged(int)", self.onRoiChanged)
        self.splitCombo.connect("currentIndexChanged(int)", self.onSplitChanged)
        self.loadButton.connect("clicked()", self.onLoad)
        self.previousButton.connect("clicked()", self.onPrevious)
        self.nextButton.connect("clicked()", self.onNext)
        self.clearButton.connect("clicked()", self.clearLoadedNodes)
        self.fitButton.connect("clicked()", self.fitViews)
        self.showCT.connect("toggled(bool)", self.updateVisibility)
        self.opacity.connect("valueChanged(double)", self.updateVisibility)
        self.thickness.connect("valueChanged(double)", self.updateVisibility)
        self.show3D.connect("toggled(bool)", self.updateVisibility)
        for cb in self.segmentChecks.values():
            cb.connect("toggled(bool)", self.updateVisibility)

        if os.path.isdir(DEFAULT_NO_CORRUPTION_ROOT) and os.path.isdir(DEFAULT_NSRQA_ROOT):
            self.onScan()

    @staticmethod
    def _colorLabel(text, color):
        r, g, b = [int(round(c * 255)) for c in color]
        label = qt.QLabel(text)
        label.setStyleSheet(f"QLabel {{ font-weight: 600; color: rgb({r},{g},{b}); }}")
        return label

    def cleanup(self):
        self.clearLoadedNodes()

    def onScan(self):
        baselineRoot = Path(self.baselineDirectoryButton.currentPath)
        nsrqaRoot = Path(self.nsrqaDirectoryButton.currentPath)
        self.caseIndex, summary = self.logic.scanPairedRoots(baselineRoot, nsrqaRoot)

        self.modeCombo.blockSignals(True)
        self.modeCombo.clear()
        for mode in sorted(self.caseIndex):
            self.modeCombo.addItem(mode)
        self.modeCombo.blockSignals(False)
        self.onModeChanged()

        self.statusLabel.text = (
            f"Paired cases: {summary['paired']}. "
            f"No-corruption only: {summary['baseline_only']}. "
            f"NSR-QA only: {summary['nsrqa_only']}."
        )

    def onModeChanged(self, *_args):
        self.roiCombo.blockSignals(True)
        self.roiCombo.clear()
        mode = self.modeCombo.currentText
        for roi in sorted(self.caseIndex.get(mode, {}), key=str.lower):
            self.roiCombo.addItem(roi)
        self.roiCombo.blockSignals(False)
        self.onRoiChanged()

    def onRoiChanged(self, *_args):
        self.splitCombo.blockSignals(True)
        self.splitCombo.clear()
        mode = self.modeCombo.currentText
        roi = self.roiCombo.currentText
        splits = self.caseIndex.get(mode, {}).get(roi, {})
        for split in ("training1", "validation1", "testing1"):
            if split in splits:
                self.splitCombo.addItem(split)
        for split in sorted(set(splits) - {"training1", "validation1", "testing1"}):
            self.splitCombo.addItem(split)
        self.splitCombo.blockSignals(False)
        self.onSplitChanged()

    def onSplitChanged(self, *_args):
        self.patientCombo.clear()
        cases = self._currentCases()
        for pid in sorted(cases, key=self.logic.patientSortKey):
            self.patientCombo.addItem(pid)
        enabled = bool(cases)
        self.loadButton.enabled = enabled
        self.previousButton.enabled = enabled
        self.nextButton.enabled = enabled

    def _currentCases(self):
        return (
            self.caseIndex
            .get(self.modeCombo.currentText, {})
            .get(self.roiCombo.currentText, {})
            .get(self.splitCombo.currentText, {})
        )

    def onPrevious(self):
        count = self.patientCombo.count
        if count <= 0:
            return
        self.patientCombo.currentIndex = (self.patientCombo.currentIndex - 1) % count
        self.onLoad()

    def onNext(self):
        count = self.patientCombo.count
        if count <= 0:
            return
        self.patientCombo.currentIndex = (self.patientCombo.currentIndex + 1) % count
        self.onLoad()

    def onLoad(self):
        mode = self.modeCombo.currentText
        roi = self.roiCombo.currentText
        split = self.splitCombo.currentText
        pid = self.patientCombo.currentText
        pair = self._currentCases().get(pid)
        if not pair:
            slicer.util.errorDisplay("No paired case found for the current selection.")
            return

        self.clearLoadedNodes()
        try:
            ct, seg, segmentIds, missing = self.logic.loadPairedCase(pair, roi, pid)
            self.loadedNodes = {"ct": ct, "seg": seg}
            self.segmentIds = segmentIds
            slicer.util.setSliceViewerLayers(background=ct)
            self.updateVisibility()
            self.fitViews()

            msg = f"Loaded {mode} / {roi} / {split} / patient {pid}."
            if missing:
                msg += " Missing: " + ", ".join(missing)
            self.statusLabel.text = msg
        except Exception as exc:
            logging.exception("Failed to load paired corruption-ablation case")
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

        fill = float(self.opacity.value)
        for key, segId in self.segmentIds.items():
            display.SetSegmentVisibility(segId, bool(self.segmentChecks[key].checked))
            try:
                display.SetSegmentOpacity2DFill(segId, fill)
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


class AutosegCorruptionComparisonViewerLogic(ScriptedLoadableModuleLogic):
    REQUIRED_CASE_FILES = (
        "ct.nii.gz",
        "approved.nii.gz",
        "autoseg_input.nii.gz",
        "nsrqa_reconstruction.nii.gz",
    )

    @staticmethod
    def patientSortKey(pid):
        try:
            return (0, int(pid))
        except Exception:
            return (1, str(pid).lower())

    @staticmethod
    def _scanSingleRoot(root: Path):
        cases = {}
        if not root.is_dir():
            return cases
        for p in root.glob("*/*/*/patient_*"):
            if not p.is_dir():
                continue
            try:
                mode, roi, split, patientDir = p.relative_to(root).parts
            except ValueError:
                continue
            if not all((p / name).is_file() for name in AutosegCorruptionComparisonViewerLogic.REQUIRED_CASE_FILES):
                continue
            pid = patientDir.replace("patient_", "", 1)
            cases[(mode, roi, split, pid)] = p
        return cases

    def scanPairedRoots(self, baselineRoot: Path, nsrqaRoot: Path):
        baseline = self._scanSingleRoot(baselineRoot)
        nsrqa = self._scanSingleRoot(nsrqaRoot)
        baselineKeys = set(baseline)
        nsrqaKeys = set(nsrqa)
        pairedKeys = baselineKeys & nsrqaKeys

        result = {}
        for mode, roi, split, pid in pairedKeys:
            result.setdefault(mode, {}).setdefault(roi, {}).setdefault(split, {})[pid] = {
                "baseline": baseline[(mode, roi, split, pid)],
                "nsrqa": nsrqa[(mode, roi, split, pid)],
            }

        summary = {
            "paired": len(pairedKeys),
            "baseline_only": len(baselineKeys - nsrqaKeys),
            "nsrqa_only": len(nsrqaKeys - baselineKeys),
        }
        return result, summary

    def _loadLabel(self, path: Path, node, name, color, reference):
        if not path.is_file():
            return None
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
                return None
            sid = newIds[-1]
            segment = seg.GetSegment(sid)
            segment.SetName(name)
            segment.SetColor(*[float(c) for c in color])
            return sid
        finally:
            slicer.mrmlScene.RemoveNode(label)

    def _newSegmentation(self, name, ct):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        node.SetName(name)
        node.CreateDefaultDisplayNodes()
        node.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        return node

    def loadPairedCase(self, pair, roi: str, pid: str):
        baselineDir = Path(pair["baseline"])
        nsrqaDir = Path(pair["nsrqa"])

        ctPath = baselineDir / "ct.nii.gz"
        if not ctPath.is_file():
            raise FileNotFoundError(ctPath)
        ct = slicer.util.loadVolume(str(ctPath))
        if not ct:
            raise RuntimeError(f"Could not load CT: {ctPath}")
        ct.SetName(f"CT-{roi}-patient_{pid}")

        seg = self._newSegmentation(f"CorruptionComparison-{roi}-patient_{pid}", ct)
        paths = {
            "approved": baselineDir / "approved.nii.gz",
            "autoseg": baselineDir / "autoseg_input.nii.gz",
            "baseline_recon": baselineDir / "nsrqa_reconstruction.nii.gz",
            "nsrqa_recon": nsrqaDir / "nsrqa_reconstruction.nii.gz",
        }

        segmentIds = {}
        missing = []
        specByKey = {key: (label, color) for key, label, color in SEGMENT_SPECS}
        for key in ("approved", "autoseg", "baseline_recon", "nsrqa_recon"):
            label, color = specByKey[key]
            sid = self._loadLabel(paths[key], seg, label, color, ct)
            if sid:
                segmentIds[key] = sid
            else:
                missing.append(str(paths[key]))

        try:
            seg.CreateClosedSurfaceRepresentation()
        except Exception:
            pass
        return ct, seg, segmentIds, missing


class AutosegCorruptionComparisonViewerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.delayDisplay("Autoseg Corruption Comparison Viewer loaded")
