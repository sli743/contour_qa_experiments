import os
from pathlib import Path
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

DEFAULT_ROOT = r"D:\aiqa\contour_QA\supervised_autoseg_correction\evaluations"
SEGMENTS = [
    ("approved", "Approved", "approved.nii.gz", (0.0, 0.8, 1.0)),
    ("autoseg", "Autoseg input", "autoseg_input.nii.gz", (1.0, 0.45, 0.0)),
    ("predicted", "Predicted approved", "predicted_approved.nii.gz", (0.3, 0.9, 0.25)),
    ("true_missing", "True missing", "true_missing.nii.gz", (0.0, 0.9, 1.0)),
    ("true_excess", "True excess", "true_excess.nii.gz", (1.0, 0.0, 0.8)),
    ("corr_added", "Correction added", "correction_added.nii.gz", (0.6, 0.3, 1.0)),
    ("corr_removed", "Correction removed", "correction_removed.nii.gz", (1.0, 0.1, 0.1)),
]

class SupervisedAutosegCorrectionViewer(ScriptedLoadableModule):
    def __init__(self,parent):
        ScriptedLoadableModule.__init__(self,parent); self.parent.title="Supervised Autoseg Correction Viewer"; self.parent.categories=["Segmentation"]
        self.parent.helpText="Inspect held-out autoseg -> predicted-approved correction cases from the 5-fold supervised QA experiment."

class SupervisedAutosegCorrectionViewerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self); self.index={}; self.nodes=[]; self.seg=None; self.segIds={}
        box=ctk.ctkCollapsibleButton(); box.text="Input"; self.layout.addWidget(box); form=qt.QFormLayout(box)
        self.root=ctk.ctkPathLineEdit(); self.root.filters=ctk.ctkPathLineEdit.Dirs; self.root.currentPath=DEFAULT_ROOT; form.addRow("Evaluation root:",self.root)
        self.scan=qt.QPushButton("Scan"); form.addRow(self.scan); self.roi=qt.QComboBox(); form.addRow("ROI:",self.roi); self.fold=qt.QComboBox(); form.addRow("Fold:",self.fold); self.patient=qt.QComboBox(); form.addRow("Patient:",self.patient)
        self.load=qt.QPushButton("Load case"); form.addRow(self.load)
        vis=ctk.ctkCollapsibleButton(); vis.text="Visibility"; self.layout.addWidget(vis); vf=qt.QFormLayout(vis)
        self.checks={}
        for key,label,_,_ in SEGMENTS:
            cb=qt.QCheckBox(); cb.checked=key in ("approved","autoseg","predicted"); self.checks[key]=cb; vf.addRow(label,cb)
        self.opacity=ctk.ctkSliderWidget(); self.opacity.minimum=0; self.opacity.maximum=1; self.opacity.value=.25; self.opacity.singleStep=.05; vf.addRow("2D fill opacity:",self.opacity)
        self.thickness=ctk.ctkSliderWidget(); self.thickness.minimum=1; self.thickness.maximum=10; self.thickness.value=2; self.thickness.decimals=0; vf.addRow("2D contour thickness:",self.thickness)
        self.show3d=qt.QCheckBox(); vf.addRow("Show 3D:",self.show3d); self.fit=qt.QPushButton("Fit slice views"); vf.addRow(self.fit)
        self.metrics=qt.QLabel("No case loaded."); self.metrics.wordWrap=True; self.layout.addWidget(self.metrics); self.layout.addStretch(1)
        self.scan.connect("clicked()",self.onScan); self.roi.connect("currentIndexChanged(int)",self.updateFolds); self.fold.connect("currentIndexChanged(int)",self.updatePatients); self.load.connect("clicked()",self.onLoad); self.opacity.connect("valueChanged(double)",self.updateVis); self.thickness.connect("valueChanged(double)",self.updateVis); self.show3d.connect("toggled(bool)",self.updateVis); self.fit.connect("clicked()",self.fitViews)
        for cb in self.checks.values(): cb.connect("toggled(bool)",self.updateVis)
        if os.path.isdir(DEFAULT_ROOT): self.onScan()
    def onScan(self):
        root=Path(self.root.currentPath); self.index={}
        for p in root.glob("*/fold_*/test_cases/patient_*"):
            if (p/"ct.nii.gz").is_file(): self.index.setdefault(p.parents[2].name,{}).setdefault(p.parents[1].name,{})[p.name.replace("patient_","")]=p
        self.roi.clear(); [self.roi.addItem(r) for r in sorted(self.index)]; self.updateFolds()
    def updateFolds(self,*_):
        self.fold.clear(); r=self.roi.currentText; [self.fold.addItem(f) for f in sorted(self.index.get(r,{}))]; self.updatePatients()
    def updatePatients(self,*_):
        self.patient.clear(); d=self.index.get(self.roi.currentText,{}).get(self.fold.currentText,{})
        for p in sorted(d,key=lambda x:int(x) if str(x).isdigit() else x): self.patient.addItem(p)
    def clear(self):
        for n in self.nodes:
            if n: slicer.mrmlScene.RemoveNode(n)
        self.nodes=[]; self.seg=None; self.segIds={}
    def onLoad(self):
        d=self.index.get(self.roi.currentText,{}).get(self.fold.currentText,{}).get(self.patient.currentText)
        if not d:return
        self.clear(); ok,ct=slicer.util.loadVolume(str(d/"ct.nii.gz"),returnNode=True); self.nodes.append(ct)
        self.seg=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode","Correction contours"); self.seg.CreateDefaultDisplayNodes(); self.nodes.append(self.seg)
        for key,label,fn,color in SEGMENTS:
            p=d/fn
            if not p.is_file():continue
            ok,node=slicer.util.loadLabelVolume(str(p),returnNode=True); self.nodes.append(node)
            sid=slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(node,self.seg)
            seg=self.seg.GetSegmentation(); ids=vtk.vtkStringArray(); seg.GetSegmentIDs(ids); sid=ids.GetValue(ids.GetNumberOfValues()-1); seg.GetSegment(sid).SetName(label); seg.GetSegment(sid).SetColor(*color); self.segIds[key]=sid; slicer.mrmlScene.RemoveNode(node); self.nodes.remove(node)
        mp=d/"metrics.json"; self.metrics.text=mp.read_text() if mp.is_file() else "metrics.json missing"; self.updateVis(); self.fitViews()
    def updateVis(self,*_):
        if not self.seg:return
        disp=self.seg.GetDisplayNode(); disp.SetOpacity2DFill(float(self.opacity.value)); disp.SetOpacity2DOutline(1.0); disp.SetVisibility3D(bool(self.show3d.checked))
        try: disp.SetSliceIntersectionThickness(int(round(self.thickness.value)))
        except Exception: pass
        for k,sid in self.segIds.items(): disp.SetSegmentVisibility(sid,bool(self.checks[k].checked)); disp.SetSegmentVisibility3D(sid,bool(self.checks[k].checked and self.show3d.checked))
    def fitViews(self): slicer.util.resetSliceViews()
    def cleanup(self): self.clear()
