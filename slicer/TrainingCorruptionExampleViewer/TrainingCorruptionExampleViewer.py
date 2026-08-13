import os
import re
import logging
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *


DEFAULT_ROOT = r"D:\aiqa\contour_QA\all_roi_NSRQA_study\training_corruption_examples"

# Ordered for consistent UI and figure construction.
CONTOUR_SPECS = [
    ("clean", "Clean target", "clean_target_patient_{pid}.nii.gz", (0.00, 0.85, 1.00)),
    ("smooth_warp", "Smooth warp", "corrupted_input_smooth_warp_patient_{pid}.nii.gz", (1.00, 0.55, 0.00)),
    ("distance_noise", "Distance noise", "corrupted_input_distance_noise_patient_{pid}.nii.gz", (0.30, 0.85, 0.25)),
    ("combined", "Combined", "corrupted_input_combined_patient_{pid}.nii.gz", (0.95, 0.10, 0.70)),
    ("dropout", "Stochastic dropout", "corrupted_input_stochastic_region_dropout_patient_{pid}.nii.gz", (0.95, 0.90, 0.05)),
    ("offset_in", "Global offset inward", "corrupted_input_global_boundary_offset_inward_patient_{pid}.nii.gz", (0.55, 0.25, 1.00)),
    ("offset_out", "Global offset outward", "corrupted_input_global_boundary_offset_outward_patient_{pid}.nii.gz", (1.00, 0.15, 0.10)),
]

AUX_SPECS = [
    ("changed", "Changed region", "changed_region_{tag}_patient_{pid}.nii.gz", (1.00, 1.00, 1.00)),
    ("missing", "Missing region", "missing_region_{tag}_patient_{pid}.nii.gz", (0.00, 0.90, 1.00)),
    ("excess", "Excess region", "excess_region_{tag}_patient_{pid}.nii.gz", (1.00, 0.00, 0.85)),
]

CORRUPTION_TAGS = [
    ("smooth_warp", "Smooth warp"),
    ("distance_noise", "Distance noise"),
    ("combined", "Combined"),
    ("stochastic_region_dropout", "Stochastic dropout"),
    ("global_boundary_offset_inward", "Global offset inward"),
    ("global_boundary_offset_outward", "Global offset outward"),
]


class TrainingCorruptionExampleViewer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Training Corruption Example Viewer"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["OpenAI"]
        self.parent.helpText = """
Inspect standardized NSR-QA training-corruption examples. The module scans the
all-ROI training_corruption_examples directory, loads the original CT as the
background volume, and overlays clean and corrupted contours with independent
visibility toggles and fixed high-contrast colors.
"""
        self.parent.acknowledgementText = "Generated for the AIQA contour-QA workflow."


class TrainingCorruptionExampleViewerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = TrainingCorruptionExampleViewerLogic()
        self.caseIndex = {}
        self.loadedNodes = {"ct": None, "seg": None, "aux": None}
        self.primarySegmentIds = {}
        self.auxSegmentIds = {}

        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Input"
        self.layout.addWidget(inputCollapsible)
        inputForm = qt.QFormLayout(inputCollapsible)

        self.directoryButton = ctk.ctkPathLineEdit()
        self.directoryButton.filters = ctk.ctkPathLineEdit.Dirs
        self.directoryButton.settingKey = "TrainingCorruptionExampleViewer/rootDir"
        self.directoryButton.currentPath = DEFAULT_ROOT
        inputForm.addRow("Examples root:", self.directoryButton)

        self.scanButton = qt.QPushButton("Scan directory")
        self.scanButton.toolTip = "Find ROI/example folders below the selected root"
        inputForm.addRow(self.scanButton)

        self.roiComboBox = qt.QComboBox()
        inputForm.addRow("ROI:", self.roiComboBox)

        self.patientComboBox = qt.QComboBox()
        inputForm.addRow("Patient:", self.patientComboBox)

        buttonRow = qt.QHBoxLayout()
        self.loadButton = qt.QPushButton("Load selected example")
        self.clearButton = qt.QPushButton("Clear loaded nodes")
        buttonRow.addWidget(self.loadButton)
        buttonRow.addWidget(self.clearButton)
        inputForm.addRow(buttonRow)

        visCollapsible = ctk.ctkCollapsibleButton()
        visCollapsible.text = "Primary contour visibility"
        self.layout.addWidget(visCollapsible)
        visForm = qt.QFormLayout(visCollapsible)

        self.showCTCheckBox = qt.QCheckBox()
        self.showCTCheckBox.checked = True
        visForm.addRow("Show CT:", self.showCTCheckBox)

        self.primaryChecks = {}
        for key, label, _pattern, color in CONTOUR_SPECS:
            cb = qt.QCheckBox()
            cb.checked = True
            self.primaryChecks[key] = cb
            visForm.addRow(self._colorLabel(label, color), cb)

        self.opacitySlider = ctk.ctkSliderWidget()
        self.opacitySlider.minimum = 0.0
        self.opacitySlider.maximum = 1.0
        self.opacitySlider.singleStep = 0.05
        self.opacitySlider.value = 0.35
        self.opacitySlider.decimals = 2
        visForm.addRow("2D fill opacity:", self.opacitySlider)

        self.show3DCheckBox = qt.QCheckBox()
        self.show3DCheckBox.checked = False
        visForm.addRow("Show contours in 3D:", self.show3DCheckBox)

        self.fitViewsButton = qt.QPushButton("Fit slice views")
        visForm.addRow(self.fitViewsButton)

        auxCollapsible = ctk.ctkCollapsibleButton()
        auxCollapsible.text = "Diagnostic masks (optional)"
        auxCollapsible.collapsed = True
        self.layout.addWidget(auxCollapsible)
        auxForm = qt.QFormLayout(auxCollapsible)

        self.auxCorruptionCombo = qt.QComboBox()
        for tag, label in CORRUPTION_TAGS:
            self.auxCorruptionCombo.addItem(label, tag)
        auxForm.addRow("Corruption:", self.auxCorruptionCombo)

        self.loadAuxButton = qt.QPushButton("Load diagnostic masks")
        auxForm.addRow(self.loadAuxButton)

        self.auxChecks = {}
        for key, label, _pattern, color in AUX_SPECS:
            cb = qt.QCheckBox()
            cb.checked = True
            self.auxChecks[key] = cb
            auxForm.addRow(self._colorLabel(label, color), cb)

        self.statusLabel = qt.QLabel("Select the examples root and scan.")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)

        self.scanButton.connect("clicked()", self.onScan)
        self.roiComboBox.connect("currentIndexChanged(int)", self.onRoiChanged)
        self.loadButton.connect("clicked()", self.onLoad)
        self.clearButton.connect("clicked()", self.clearLoadedNodes)
        self.showCTCheckBox.connect("toggled(bool)", self.updateVisibility)
        self.show3DCheckBox.connect("toggled(bool)", self.updateVisibility)
        self.opacitySlider.connect("valueChanged(double)", self.updateVisibility)
        self.fitViewsButton.connect("clicked()", self.fitViews)
        self.loadAuxButton.connect("clicked()", self.onLoadAux)
        for cb in self.primaryChecks.values():
            cb.connect("toggled(bool)", self.updateVisibility)
        for cb in self.auxChecks.values():
            cb.connect("toggled(bool)", self.updateVisibility)

        # Scan immediately when the default folder exists on this machine.
        if os.path.isdir(DEFAULT_ROOT):
            self.onScan()

    def cleanup(self):
        self.clearLoadedNodes()

    @staticmethod
    def _colorLabel(text, color):
        r, g, b = [int(round(c * 255)) for c in color]
        label = qt.QLabel(text)
        label.setStyleSheet(
            "QLabel { font-weight: 600; "
            f"color: rgb({r},{g},{b}); }}"
        )
        return label

    def onScan(self):
        root = Path(self.directoryButton.currentPath)
        self.caseIndex = self.logic.scanRoot(root)
        self.roiComboBox.blockSignals(True)
        self.roiComboBox.clear()
        for roi in sorted(self.caseIndex.keys(), key=str.lower):
            self.roiComboBox.addItem(roi)
        self.roiComboBox.blockSignals(False)
        self.onRoiChanged()
        ncases = sum(len(v) for v in self.caseIndex.values())
        self.statusLabel.text = f"Found {len(self.caseIndex)} ROIs / {ncases} standardized example case(s)."

    def onRoiChanged(self, *_args):
        self.patientComboBox.clear()
        roi = self.roiComboBox.currentText
        cases = self.caseIndex.get(roi, {})
        for pid in sorted(cases.keys(), key=self.logic.patientSortKey):
            self.patientComboBox.addItem(str(pid))
        self.loadButton.enabled = bool(cases)

    def onLoad(self):
        roi = self.roiComboBox.currentText
        pid = self.patientComboBox.currentText
        caseDir = self.caseIndex.get(roi, {}).get(pid)
        if not caseDir:
            slicer.util.errorDisplay("No standardized example found for the selected ROI/patient.")
            return
        self.clearLoadedNodes()
        try:
            ctNode, segNode, segmentIds, missing = self.logic.loadPrimaryCase(caseDir, roi, pid)
            self.loadedNodes["ct"] = ctNode
            self.loadedNodes["seg"] = segNode
            self.primarySegmentIds = segmentIds
            self.updateVisibility()
            if ctNode:
                slicer.util.setSliceViewerLayers(background=ctNode)
            self.fitViews()
            if missing:
                self.statusLabel.text = (
                    f"Loaded {roi}, patient {pid}. Missing optional/expected files: " + ", ".join(missing)
                )
            else:
                self.statusLabel.text = f"Loaded {roi}, patient {pid}: CT + {len(segmentIds)} primary contours."
        except Exception as exc:
            logging.exception("Failed to load training corruption example")
            slicer.util.errorDisplay(str(exc))

    def onLoadAux(self):
        roi = self.roiComboBox.currentText
        pid = self.patientComboBox.currentText
        caseDir = self.caseIndex.get(roi, {}).get(pid)
        if not caseDir:
            return
        tag = self.auxCorruptionCombo.currentData
        if not tag:
            return
        if self.loadedNodes.get("aux"):
            slicer.mrmlScene.RemoveNode(self.loadedNodes["aux"])
            self.loadedNodes["aux"] = None
            self.auxSegmentIds = {}
        try:
            node, segmentIds, missing = self.logic.loadAuxMasks(caseDir, roi, pid, tag, self.loadedNodes.get("ct"))
            self.loadedNodes["aux"] = node
            self.auxSegmentIds = segmentIds
            self.updateVisibility()
            msg = f"Loaded diagnostic masks for {tag}."
            if missing:
                msg += " Missing: " + ", ".join(missing)
            self.statusLabel.text = msg
        except Exception as exc:
            logging.exception("Failed to load diagnostic masks")
            slicer.util.errorDisplay(str(exc))

    def updateVisibility(self, *_args):
        ct = self.loadedNodes.get("ct")
        if ct and ct.GetDisplayNode():
            ct.GetDisplayNode().SetVisibility(bool(self.showCTCheckBox.checked))

        segNode = self.loadedNodes.get("seg")
        if segNode and segNode.GetDisplayNode():
            display = segNode.GetDisplayNode()
            display.SetVisibility(True)
            display.SetVisibility2D(True)
            display.SetVisibility3D(bool(self.show3DCheckBox.checked))
            fillOpacity = float(self.opacitySlider.value)
            for key, segId in self.primarySegmentIds.items():
                visible = bool(self.primaryChecks[key].checked)
                display.SetSegmentVisibility(segId, visible)
                try:
                    display.SetSegmentOpacity2DFill(segId, fillOpacity)
                    display.SetSegmentOpacity2DOutline(segId, 1.0)
                    display.SetSegmentOpacity3D(segId, 0.55)
                except Exception:
                    pass

        auxNode = self.loadedNodes.get("aux")
        if auxNode and auxNode.GetDisplayNode():
            display = auxNode.GetDisplayNode()
            display.SetVisibility(True)
            display.SetVisibility2D(True)
            display.SetVisibility3D(bool(self.show3DCheckBox.checked))
            for key, segId in self.auxSegmentIds.items():
                visible = bool(self.auxChecks[key].checked)
                display.SetSegmentVisibility(segId, visible)
                try:
                    display.SetSegmentOpacity2DFill(segId, 0.25)
                    display.SetSegmentOpacity2DOutline(segId, 1.0)
                    display.SetSegmentOpacity3D(segId, 0.4)
                except Exception:
                    pass

    def fitViews(self):
        try:
            slicer.util.resetSliceViews()
        except Exception:
            lm = slicer.app.layoutManager()
            for name in ("Red", "Yellow", "Green"):
                widget = lm.sliceWidget(name) if lm else None
                if widget:
                    widget.sliceLogic().FitSliceToAll()

    def clearLoadedNodes(self):
        for key in ("aux", "seg", "ct"):
            node = self.loadedNodes.get(key)
            if node:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
                self.loadedNodes[key] = None
        self.primarySegmentIds = {}
        self.auxSegmentIds = {}


class TrainingCorruptionExampleViewerLogic(ScriptedLoadableModuleLogic):
    PATIENT_RE = re.compile(r"patient_(.+?)\.nii(?:\.gz)?$", re.IGNORECASE)

    @staticmethod
    def patientSortKey(pid):
        try:
            return (0, int(pid))
        except Exception:
            return (1, str(pid).lower())

    def extractPatientId(self, name):
        match = self.PATIENT_RE.search(name)
        return match.group(1) if match else None

    def scanRoot(self, root: Path):
        """Return {roi: {patient_id: case_dir}}.

        Supports both layouts:
          root/<ROI>/<files>
          root/<ROI>/patient_<id>/<files>
        and falls back to recursive discovery of clean_target files.
        """
        root = Path(root)
        if not root.is_dir():
            return {}
        result = {}
        for cleanPath in root.rglob("clean_target_patient_*.nii.gz"):
            pid = self.extractPatientId(cleanPath.name)
            if pid is None:
                continue
            caseDir = cleanPath.parent
            try:
                rel = caseDir.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if not parts:
                continue
            roi = parts[0]
            result.setdefault(roi, {})[pid] = caseDir
        return result

    def _loadLabelAsSingleSegment(self, path, segmentationNode, segmentName, color, referenceNode=None):
        if not path.is_file():
            return None
        labelNode = slicer.util.loadLabelVolume(str(path))
        if not labelNode:
            raise RuntimeError(f"Could not load labelmap: {path}")
        try:
            before = {
                segmentationNode.GetSegmentation().GetNthSegmentID(i)
                for i in range(segmentationNode.GetSegmentation().GetNumberOfSegments())
            }
            if referenceNode is not None:
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceNode)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentationNode)
            segmentation = segmentationNode.GetSegmentation()
            after = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]
            newIds = [sid for sid in after if sid not in before]
            if not newIds:
                raise RuntimeError(f"No foreground segment imported from {path}")
            segId = newIds[-1]
            segment = segmentation.GetSegment(segId)
            segment.SetName(segmentName)
            segment.SetColor(*[float(c) for c in color])
            return segId
        finally:
            slicer.mrmlScene.RemoveNode(labelNode)

    def loadPrimaryCase(self, caseDir, roi, pid):
        caseDir = Path(caseDir)
        ctPath = caseDir / f"original_ct_patient_{pid}.nii.gz"
        if not ctPath.is_file():
            raise FileNotFoundError(f"Original CT is required but missing: {ctPath}")
        ctNode = slicer.util.loadVolume(str(ctPath))
        if not ctNode:
            raise RuntimeError(f"Could not load CT: {ctPath}")
        ctNode.SetName(f"CT-{roi}-patient_{pid}")

        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segNode.SetName(f"TrainingCorruptions-{roi}-patient_{pid}")
        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(ctNode)

        ids = {}
        missing = []
        for key, label, pattern, color in CONTOUR_SPECS:
            path = caseDir / pattern.format(pid=pid)
            if not path.is_file():
                missing.append(path.name)
                continue
            segId = self._loadLabelAsSingleSegment(path, segNode, label, color, ctNode)
            if segId:
                ids[key] = segId

        if not ids:
            slicer.mrmlScene.RemoveNode(segNode)
            slicer.mrmlScene.RemoveNode(ctNode)
            raise RuntimeError(f"No contour files found in {caseDir}")

        try:
            segNode.CreateClosedSurfaceRepresentation()
        except Exception as exc:
            logging.warning(f"Could not create 3D surfaces: {exc}")
        return ctNode, segNode, ids, missing

    def loadAuxMasks(self, caseDir, roi, pid, tag, ctNode=None):
        caseDir = Path(caseDir)
        if ctNode is None:
            ctPath = caseDir / f"original_ct_patient_{pid}.nii.gz"
            ctNode = slicer.util.loadVolume(str(ctPath)) if ctPath.is_file() else None

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        node.SetName(f"Diagnostics-{tag}-{roi}-patient_{pid}")
        node.CreateDefaultDisplayNodes()
        if ctNode:
            node.SetReferenceImageGeometryParameterFromVolumeNode(ctNode)

        ids, missing = {}, []
        for key, label, pattern, color in AUX_SPECS:
            path = caseDir / pattern.format(tag=tag, pid=pid)
            if not path.is_file():
                missing.append(path.name)
                continue
            segId = self._loadLabelAsSingleSegment(path, node, label, color, ctNode)
            if segId:
                ids[key] = segId
        if not ids:
            slicer.mrmlScene.RemoveNode(node)
            raise FileNotFoundError(f"No diagnostic masks found for {tag} in {caseDir}")
        try:
            node.CreateClosedSurfaceRepresentation()
        except Exception:
            pass
        return node, ids, missing


class TrainingCorruptionExampleViewerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.delayDisplay("Training Corruption Example Viewer loaded")
