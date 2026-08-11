"""Small 3D Slicer module for stochastic-region-dropout preview folders."""
import os
import re
from collections import defaultdict

import ctk, qt, slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget


class StochasticDropoutPreview(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Stochastic Dropout Preview"
        parent.categories = ["Segmentation"]
        parent.dependencies = ["Segmentations"]


class StochasticDropoutPreviewWidget(ScriptedLoadableModuleWidget):
    CORR_RE = re.compile(r"^corrupted_input_branch_(.+)_sample_(\d+)_patient_(.+)\.nii\.gz$")

    def setup(self):
        super().setup()
        self.nodes = []
        self.inventory = defaultdict(dict)

        box = ctk.ctkCollapsibleButton(); box.text = "Preview folder"; self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.path = ctk.ctkPathLineEdit(); self.path.filters = ctk.ctkPathLineEdit.Dirs
        self.path.settingKey = "StochasticDropoutPreview/folder"
        form.addRow("Folder:", self.path)
        self.scanButton = qt.QPushButton("Scan")
        form.addRow(self.scanButton)
        self.caseCombo = qt.QComboBox(); form.addRow("Case:", self.caseCombo)
        row = qt.QHBoxLayout()
        self.loadButton = qt.QPushButton("Load")
        self.clearButton = qt.QPushButton("Clear")
        row.addWidget(self.loadButton); row.addWidget(self.clearButton); form.addRow(row)
        self.zoomCheck = qt.QCheckBox(); self.zoomCheck.checked = True
        form.addRow("Fit/zoom to contours:", self.zoomCheck)
        self.show3D = qt.QCheckBox(); self.show3D.checked = False
        form.addRow("Show contours in 3D:", self.show3D)
        self.status = qt.QLabel("Select a collected preview folder."); self.status.wordWrap = True
        form.addRow(self.status)
        self.layout.addStretch(1)

        self.scanButton.connect("clicked(bool)", self.scan)
        self.loadButton.connect("clicked(bool)", self.load)
        self.clearButton.connect("clicked(bool)", self.clear)
        self.show3D.connect("toggled(bool)", self.update3D)
        if self.path.currentPath:
            self.scan()

    def scan(self):
        root = self.path.currentPath
        self.inventory.clear(); self.caseCombo.clear()
        if not root or not os.path.isdir(root):
            slicer.util.errorDisplay("Choose a valid folder."); return
        for name in sorted(os.listdir(root)):
            m = self.CORR_RE.match(name)
            if not m: continue
            branch, sample, pid = m.groups()
            key = f"patient {pid} | {branch} | sample {sample}"
            rec = self.inventory[key]
            rec["pid"] = pid; rec["branch"] = branch; rec["sample"] = sample
            rec["corrupted"] = os.path.join(root, name)
            stem = f"branch_{branch}_sample_{sample}_patient_{pid}.nii.gz"
            for kind, prefix in [("missing", "missing_region_"), ("changed", "changed_region_")]:
                path = os.path.join(root, prefix + stem)
                if os.path.exists(path): rec[kind] = path
            rec["clean"] = os.path.join(root, f"clean_input_patient_{pid}.nii.gz")
            rec["ct"] = os.path.join(root, f"original_ct_patient_{pid}.nii.gz")
        self.caseCombo.addItems(sorted(self.inventory))
        self.status.text = f"Found {len(self.inventory)} preview cases."

    def clear(self):
        for node in self.nodes:
            if node and slicer.mrmlScene.IsNodePresent(node): slicer.mrmlScene.RemoveNode(node)
        self.nodes = []

    def segmentation(self, path, name, color):
        label = slicer.util.loadLabelVolume(path, {"name": name + "_label"})
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        seg.CreateDefaultDisplayNodes(); seg.SetReferenceImageGeometryParameterFromVolumeNode(label)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, seg)
        slicer.mrmlScene.RemoveNode(label)
        display = seg.GetDisplayNode(); display.SetOpacity2DFill(0.0); display.SetOpacity2DOutline(1.0)
        display.SetVisibility3D(self.show3D.checked)
        for i in range(seg.GetSegmentation().GetNumberOfSegments()):
            segment = seg.GetSegmentation().GetSegment(seg.GetSegmentation().GetNthSegmentID(i))
            segment.SetColor(color); segment.SetName(name)
        if self.show3D.checked: seg.CreateClosedSurfaceRepresentation()
        self.nodes.append(seg); return seg

    def load(self):
        key = self.caseCombo.currentText
        if not key: return
        rec = self.inventory[key]
        missing = [k for k in ["ct", "clean", "corrupted"] if not os.path.exists(rec.get(k, ""))]
        if missing:
            slicer.util.errorDisplay("Missing: " + ", ".join(missing)); return
        self.clear()
        ct = slicer.util.loadVolume(rec["ct"], {"name": f"CT_{rec['pid']}"}); self.nodes.append(ct)
        slicer.util.setSliceViewerLayers(background=ct)
        self.segmentation(rec["clean"], "Clean", (0.1, 0.9, 0.1))
        self.segmentation(rec["corrupted"], "Corrupted", (0.95, 0.75, 0.1))
        if "missing" in rec: self.segmentation(rec["missing"], "Missing region", (0.2, 0.5, 1.0))
        if "changed" in rec: self.segmentation(rec["changed"], "Changed region", (0.9, 0.2, 0.8))
        if self.zoomCheck.checked:
            slicer.util.resetSliceViews()
        self.status.text = "Loaded " + key

    def update3D(self, checked):
        for node in self.nodes:
            if node and node.IsA("vtkMRMLSegmentationNode"):
                if checked: node.CreateClosedSurfaceRepresentation()
                node.GetDisplayNode().SetVisibility3D(bool(checked))
