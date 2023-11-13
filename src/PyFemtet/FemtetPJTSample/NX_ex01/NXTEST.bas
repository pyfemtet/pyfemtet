Attribute VB_Name = "FemtetMacro"
Option Explicit

Dim Femtet As New CFemtet
Dim Als As CAnalysis
Dim BodyAttr As CBodyAttribute
Dim Bnd As CBoundary
Dim Mtl As CMaterial
Dim Gaudi As CGaudi
Dim Gogh As CGogh
'/////////////////////////null***の説明/////////////////////////
'//下記の四つの変数はCGaudiクラスMulti***を使用する場合に用います。
'//例えばMultiFilletを使用する場合に引数であるVertex(点)は指定せず
'//複数のEdge(線)だけをFilletする場合にnullVertex()を用います。
'//「Gaudi.MultiFillet nullVertex,Edge」とすれば
'//複数のEdgeだけFilletすることができます。
'/////////////////////////////////////////////////////////////
Global nullVertex() As CGaudiVertex
Global nullEdge() As CGaudiEdge
Global nullFace() As CGaudiFace
Global nullBody() As CGaudiBody

'///////////////////////////////////////////////////

'変数の宣言
Private pi as Double
'///////////////////////////////////////////////////


'////////////////////////////////////////////////////////////
'    Main関数
'////////////////////////////////////////////////////////////
Sub FemtetMain() 
    '------- Femtet自動起動 (不要な場合やExcelで実行しない場合は下行をコメントアウトしてください) -------
    Workbooks("FemtetRef.xla").AutoExecuteFemtet

    '------- 新規プロジェクト -------
    If Femtet.OpenNewProject() = False Then
        Femtet.ShowLastError
    End If

    '------- 変数の定義 -------
    InitVariables

    '------- データベースの設定 -------
    AnalysisSetUp
    BodyAttributeSetUp
    MaterialSetUp
    BoundarySetUp

    '------- モデルの作成 -------
    Set Gaudi = Femtet.Gaudi
    MakeModel

    '------- 標準メッシュサイズの設定 -------
    '<<<<<<< 自動計算に設定する場合は-1を設定してください >>>>>>>
    Gaudi.MeshSize = 1

    '------- プロジェクトの保存 -------
    Dim ProjectFilePath As String
    ProjectFilePath = "C:\Users\mm11592\AppData\Local\Temp\FemtetPrjTmp_PID[9160]\新規プロジェクト"
    '<<<<<<< プロジェクトを保存する場合は以下のコメントを外してください >>>>>>>
    'If Femtet.SaveProject(ProjectFilePath & ".femprj", True) = False Then
    '    Femtet.ShowLastError
    'End If

    '------- メッシュの生成 -------
    '<<<<<<< メッシュを生成する場合は以下のコメントを外してください >>>>>>>
    'Gaudi.Mesh

    '------- 解析の実行 -------
    '<<<<<<< 解析を実行する場合は以下のコメントを外してください >>>>>>>
    'Femtet.Solve

    '------- 解析結果の抽出 -------
    '<<<<<<< 計算結果を抽出する場合は以下のコメントを外してください >>>>>>>
    'SamplingResult

    '------- 計算結果の保存 -------
    '<<<<<<< 計算結果(.pdt)ファイルを保存する場合は以下のコメントを外してください >>>>>>>
    'If Femtet.SavePDT(Femtet.ResultFilePath & ".pdt", True) = False Then
    '    Femtet.ShowLastError
    'End If

End Sub

'////////////////////////////////////////////////////////////
'    解析条件の設定
'////////////////////////////////////////////////////////////
Sub AnalysisSetUp()

    '------- 変数にオブジェクトの設定 -------
    Set Als = Femtet.Analysis

    '------- 解析条件共通(Common) -------
    Als.AnalysisType = GALILEO_C

    '------- 応力(Galileo) -------
    Als.Galileo.bResultDetail = True

    '------- 調和解析(Harmonic) -------
    Als.Harmonic.FreqSweepType = LINEAR_INTERVAL_C

    '------- 過渡解析(Transient) -------
    Als.Transient.bAuto = False

    '------- 熱荷重(ThermalStress) -------
    Als.ThermalStress.Temp = (25)
    Als.ThermalStress.TempRef =(25)

    '------- 高度な設定(HighLevel) -------
    Als.HighLevel.MemoryLimit = (32)

    '------- メッシュの設定(MeshProperty) -------
    Als.MeshProperty.bMiddleNode = False
    Als.MeshProperty.bAdaptiveMeshOnCurve = True
    Als.MeshProperty.bAutoCalcAutoAirMeshSize = False
    Als.MeshProperty.AutoAirMeshSize = (60)
    Als.MeshProperty.bChangePlane = True
    Als.MeshProperty.bMeshG2 = True
    Als.MeshProperty.bPeriodMesh = False

    '------- 複数ステップ設定(StepAnalysis) -------
    Als.StepAnalysis.Set_Table_withoutTime 0, (20), (2.5) * 10 ^ (1)

    '------- 結果インポート(Import) -------
    Als.Import.AnalysisModelName = "未選択"
End Sub

'////////////////////////////////////////////////////////////
'    Body属性全体の設定
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp()

    '------- 変数にオブジェクトの設定 -------
    Set BodyAttr = Femtet.BodyAttribute

    '------- Body属性の設定 -------
    BodyAttributeSetUp_ボディ属性_001
End Sub

'////////////////////////////////////////////////////////////
'    Body属性の設定 Body属性名：ボディ属性_001
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_ボディ属性_001()
    '------- Body属性のIndexを保存する変数 -------
    Dim Index As Integer

    '------- Body属性の追加 -------
    BodyAttr.Add "ボディ属性_001" 

    '------- Body属性 Indexの設定 -------
    Index = BodyAttr.Ask ( "ボディ属性_001" ) 

    '------- シートボディの厚み or 2次元解析の奥行き(BodyThickness)/ワイヤーボディ幅(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = True

    '------- 方向(Direction) -------
    BodyAttr.Direction(Index).SetAxisVector (0), (0), (1)

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = True

    '------- 輻射(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Material全体の設定
'////////////////////////////////////////////////////////////
Sub MaterialSetUp()

    '------- 変数にオブジェクトの設定 -------
    Set Mtl = Femtet.Material

    '------- Materialの設定 -------
    MaterialSetUp_007_鉄Fe
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：007_鉄Fe
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_007_鉄Fe()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "007_鉄Fe" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "007_鉄Fe" ) 

    '------- 透磁率(Permeability) -------
    Mtl.Permeability(Index).sMu = (5000)

    '------- 抵抗率(Resistivity) -------
    Mtl.Resistivity(Index).sRho = (9.71) * 10 ^ (-8)
    Mtl.Resistivity(Index).bSpline = False
    Mtl.Resistivity(Index).Set_Table 0, (-195), (0.07) * 10 ^ (-7)
    Mtl.Resistivity(Index).Set_Table 1, (0), (0.89) * 10 ^ (-7)
    Mtl.Resistivity(Index).Set_Table 2, (100), (1.47) * 10 ^ (-7)
    Mtl.Resistivity(Index).Set_Table 3, (300), (3.15) * 10 ^ (-7)
    Mtl.Resistivity(Index).Set_Table 4, (700), (8.55) * 10 ^ (-7)

    '------- 導電率(ElectricConductivity) -------
    Mtl.ElectricConductivity(Index).ConductorType = CONDUCTOR_C
    Mtl.ElectricConductivity(Index).sSigma = (1.03) * 10 ^ (7)

    '------- 比熱(SpecificHeat) -------
    Mtl.SpecificHeat(Index).C = (451.786193929627)
    Mtl.SpecificHeat(Index).bSpline = False
    Mtl.SpecificHeat(Index).Set_Table 0, (-173), (0.2157758080401) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 1, (-73), (0.3842779120781) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 2, (20), (0.4517861939296) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 3, (127), (0.4906437460829) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 4, (327), (0.5658519115409) * 10 ^ (3)

    '------- 密度(Density) -------
    Mtl.Density(Index).Dens = (7874)

    '------- 熱伝導率(ThermalConductivity) -------
    Mtl.ThermalConductivity(Index).sRmd = (80.3)
    Mtl.ThermalConductivity(Index).bSpline = False
    Mtl.ThermalConductivity(Index).Set_Table 0, (-173), (132)
    Mtl.ThermalConductivity(Index).Set_Table 1, (-73), (94)
    Mtl.ThermalConductivity(Index).Set_Table 2, (27), (80.3)
    Mtl.ThermalConductivity(Index).Set_Table 3, (127), (69.4)

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (1.18) * 10 ^ (-5)
    Mtl.Expansion(Index).bSpline = False
    Mtl.Expansion(Index).Set_Table 0, (-173), (0.56) * 10 ^ (-5)
    Mtl.Expansion(Index).Set_Table 1, (20), (1.18) * 10 ^ (-5)
    Mtl.Expansion(Index).Set_Table 2, (227), (1.44) * 10 ^ (-5)
    Mtl.Expansion(Index).Set_Table 3, (527), (1.62) * 10 ^ (-5)

    '------- 弾性定数(Elasticity) -------
    Mtl.Elasticity(Index).sY = (206) * 10 ^ (9)
    Mtl.Elasticity(Index).Nu = (0.28)

    '------- 圧電定数(PiezoElectricity) -------
    Mtl.PiezoElectricity(Index).bPiezo = False

    '------- 説明 -------
    Mtl.Comment(Index).Comment = "《出典》 " & Chr(13) & Chr(10) & "　密度： [C2]P.26" & Chr(13) & Chr(10) & "　弾性定数： [C2]P.26" & Chr(13) & Chr(10) & "　線膨張係数： [S]P.484" & Chr(13) & Chr(10) & "　抵抗率： [C2]P.490, [S]P.527" & Chr(13) & Chr(10) & "　比熱： [S]P.473" & Chr(13) & Chr(10) & "　熱伝導率： [C2]P.70" & Chr(13) & Chr(10) & "　　" & Chr(13) & Chr(10) & "《参考文献》　" & Chr(13) & Chr(10) & "　[C2] 化学便覧 基礎編II 改訂4版 日本化学学会編 丸善(1993)" & Chr(13) & Chr(10) & "　[S] 理科年表 平成8年 国立天文台編 丸善(1996)"
End Sub

'////////////////////////////////////////////////////////////
'    Boundary全体の設定
'////////////////////////////////////////////////////////////
Sub BoundarySetUp()

    '------- 変数にオブジェクトの設定 -------
    Set Bnd = Femtet.Boundary

    '------- Boundaryの設定 -------
    BoundarySetUp_RESERVED_default
    BoundarySetUp_fix
    BoundarySetUp_load
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：RESERVED_default (外部境界条件)
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_RESERVED_default()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "RESERVED_default" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "RESERVED_default" ) 

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = True
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：fix
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_fix()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "fix" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "fix" ) 

    '------- 機械(Mechanical) -------
    Bnd.Mechanical(Index).Condition = DISPLACEMENT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = True
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：load
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_load()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "load" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "load" ) 

    '------- 機械(Mechanical) -------
    Bnd.Mechanical(Index).Condition = FACE_LOAD_C
    Bnd.Mechanical(Index).SetT (0), (0), (-10)
    Bnd.Mechanical(Index).SetTM (0), (0), (0)
    Bnd.Mechanical(Index).bT = True

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = True
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    IF関数
'////////////////////////////////////////////////////////////
Function F_IF(expression As Double, val_true As Double, val_false As Double) As Double
    If expression Then
        F_IF = val_true
    Else
        F_IF = val_false
    End If

End Function

'////////////////////////////////////////////////////////////
'    変数定義関数
'////////////////////////////////////////////////////////////
Sub InitVariables()


    'VB上の変数の定義
    pi = 3.1415926535897932

    'FemtetGUI上の変数の登録（既存モデルの変数制御等でのみ利用）

End Sub

'////////////////////////////////////////////////////////////
'    モデル作成関数
'////////////////////////////////////////////////////////////
Sub MakeModel()

    '------- Body配列変数の定義 -------
    Dim Body() as CGaudiBody

    '------- モデルを描画させない設定 -------
    Femtet.RedrawMode = False


    '------- Import2 -------
    ReDim Preserve Body(0)
    Dim BodyArray0() As CGaudiBody
    Gaudi.Import2 "C:\Users\mm11592\Documents\myFiles2\working\PyFemtetOpt2\PyFemtetOptGit\NXTEST.x_t", True, BodyArray0, False
    Set Body(0) = BodyArray0(0)

    '------- SetName -------
    Body(0).SetName "ボディ属性_001", "007_鉄Fe"

    '------- SetBoundary -------
    Dim Face0(1) As CGaudiFace
    Set Face0(0) = Body(0).GetFaceByID(315)
    Set Face0(1) = Body(0).GetFaceByID(329)
    Gaudi.MultiSetBoundary nullVertex, nullEdge, Face0, nullBody, "fix"

    '------- SetBoundary -------
    Dim Face1 As CGaudiFace
    Set Face1 = Body(0).GetFaceByID(317)
    Face1.SetBoundary "load"


    '------- モデルを再描画します -------
    Femtet.Redraw

End Sub

'////////////////////////////////////////////////////////////
'    計算結果抽出関数
'////////////////////////////////////////////////////////////
Sub SamplingResult()

    '------- 変数にオブジェクトの設定 -------
    Set Gogh = Femtet.Gogh

    '------- 現在の計算結果を中間ファイルから開く -------
    If Femtet.OpenCurrentResult(True) = False Then
        Femtet.ShowLastError
    End If

    '------- フィールドの設定 -------
    Gogh.Galileo.Vector = GALILEO_DISPLACEMENT_C

    '------- 最大値の取得 -------
    Dim PosMax() As Double '最大値の座標
    Dim ResultMax As Double ' 最大値

    If Gogh.Galileo.GetMAXVectorPoint(VEC_C, CMPX_REAL_C, PosMax, ResultMax) = False Then
        Femtet.ShowLastError
    End If

    '------- 最小値の取得 -------
    Dim PosMin() As Double '最小値の座標
    Dim ResultMin As Double '最小値

    If Gogh.Galileo.GetMINVectorPoint(VEC_C, CMPX_REAL_C, PosMin, ResultMin) = False Then
        Femtet.ShowLastError
    End If

    '------- 任意座標の計算結果の取得 -------
    Dim Value() As New CComplex

    If Gogh.Galileo.GetVectorAtPoint(0, 0, 0, Value()) = False Then
        Femtet.ShowLastError
    End If

    ' 複数の座標の結果をまとめて取得する場合は、MultiGetVectorAtPoint関数をご利用ください。

End Sub

