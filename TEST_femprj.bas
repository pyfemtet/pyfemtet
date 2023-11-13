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
Private c_pi as Double
Private r as Double
Private h as Double
Private p as Double
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
    Gaudi.MeshSize = -1

    '------- プロジェクトの保存 -------
    Dim ProjectFilePath As String
    ProjectFilePath = "C:\Users\mm11592\Documents\myFiles2\working\1_PyFemtetOpt\PyFemtetOptDevelopment\PyFemtetOptProject\src\PyFemtet\FemtetPJTSample\TEST_NX\TEST_femprj"
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
    Als.AnalysisType = PASCAL_C

    '------- 磁場(Gauss) -------
    Als.Gauss.b2ndEdgeElement = True

    '------- 応力(Galileo) -------
    Als.Galileo.PenetrateTolerance = (1.0) * 10 ^ (-3)

    '------- 電磁波(Hertz) -------
    Als.Hertz.b2ndEdgeElement = True

    '------- 圧電(Rayleigh) -------
    Als.Rayleigh.bConstantTemp = True

    '------- 高度な設定(HighLevel) -------
    Als.HighLevel.nNonL = (20)
    Als.HighLevel.bATS = False
    Als.HighLevel.FactorType = RADIO_ANALYTICAL_C
    Als.HighLevel.bUseDeathMaterial = False

    '------- メッシュの設定(MeshProperty) -------
    Als.MeshProperty.AdaptiveTolPort = (1.0) * 10 ^ (-2)
    Als.MeshProperty.AutoAirMeshSize = (180.0)
    Als.MeshProperty.bChangePlane = True
    Als.MeshProperty.bMeshG2 = True
    Als.MeshProperty.bPeriodMesh = False

    '------- 電磁界(Volta) -------
    Als.Volta.b2ndEdgeElement = True

    '------- 複数ステップ設定(StepAnalysis) -------
    Als.StepAnalysis.bSetTime = True
    Als.StepAnalysis.Set_Table_withTime 0, (1.0), (20), (0.0)
    Als.StepAnalysis.BreakStep = (100)
End Sub

'////////////////////////////////////////////////////////////
'    Body属性全体の設定
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp()

    '------- 変数にオブジェクトの設定 -------
    Set BodyAttr = Femtet.BodyAttribute

    '------- Body属性の設定 -------
    BodyAttributeSetUp_Air
    BodyAttributeSetUp_固体
End Sub

'////////////////////////////////////////////////////////////
'    Body属性の設定 Body属性名：Air
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Air()
    '------- Body属性のIndexを保存する変数 -------
    Dim Index As Integer

    '------- Body属性の追加 -------
    BodyAttr.Add "Air" 

    '------- Body属性 Indexの設定 -------
    Index = BodyAttr.Ask ( "Air" ) 

    '------- シートボディの厚み or 2次元解析の奥行き(BodyThickness)/ワイヤーボディ幅(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False

    '------- 固体/流体(SolidLiquidGas) -------
    BodyAttr.SolidLiquidGas(Index).State = GAS_C

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- 輻射(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Body属性の設定 Body属性名：固体
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_固体()
    '------- Body属性のIndexを保存する変数 -------
    Dim Index As Integer

    '------- Body属性の追加 -------
    BodyAttr.Add "固体" 

    '------- Body属性 Indexの設定 -------
    Index = BodyAttr.Ask ( "固体" ) 

    '------- シートボディの厚み or 2次元解析の奥行き(BodyThickness)/ワイヤーボディ幅(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

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
    MaterialSetUp_固体
    MaterialSetUp_000_空気

    '+++++++++++++++++++++++++++++++++++++++++
    '++使用されていないMaterialデータです
    '++使用する際はコメントを外して下さい
    '+++++++++++++++++++++++++++++++++++++++++
    'MaterialSetUp_001_アルミナ
    'MaterialSetUp_006_ガラスエポキシ
    'MaterialSetUp_008_銅Cu
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：001_アルミナ
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_001_アルミナ()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "001_アルミナ" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "001_アルミナ" ) 

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (0.0)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- 粘度(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- 着磁(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：006_ガラスエポキシ
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_006_ガラスエポキシ()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "006_ガラスエポキシ" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "006_ガラスエポキシ" ) 

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (0.0)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- 粘度(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- 着磁(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：008_銅Cu
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_008_銅Cu()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "008_銅Cu" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "008_銅Cu" ) 

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (0.0)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- 粘度(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- 着磁(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：固体
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_固体()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "固体" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "固体" ) 

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (0.0)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- 粘度(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- 着磁(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：000_空気
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_000_空気()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "000_空気" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "000_空気" ) 

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (0.0)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- 粘度(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- 固体/流体(SolidFluid) -------
    Mtl.SolidFluid(Index).State = FLUID_C

    '------- 着磁(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary全体の設定
'////////////////////////////////////////////////////////////
Sub BoundarySetUp()

    '------- 変数にオブジェクトの設定 -------
    Set Bnd = Femtet.Boundary

    '------- Boundaryの設定 -------
    BoundarySetUp_RESERVED_default
    BoundarySetUp_in
    BoundarySetUp_out
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
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)
    Bnd.FluidBern(Index).bSpline = True

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：in
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_in()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "in" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "in" ) 

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(Fluid) -------
    Bnd.Fluid(Index).Condition = VELOCITY_POTENTIAL_C
    Bnd.Fluid(Index).Vel = (1)
    Bnd.Fluid(Index).VP = (p)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)
    Bnd.FluidBern(Index).bSpline = True

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：out
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_out()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "out" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "out" ) 

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(Fluid) -------
    Bnd.Fluid(Index).Condition = VELOCITY_POTENTIAL_C

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)
    Bnd.FluidBern(Index).bSpline = True

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
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
    c_pi = 3.1415926535897932
    r = 50
    h = 50
    p = 0.5

    'FemtetGUI上の変数の登録（既存モデルの変数制御等でのみ利用）
    'Femtet.AddNewVariable "c_pi", 3.14159265e+00
    'Femtet.AddNewVariable "r", 5.00000000e+01
    'Femtet.AddNewVariable "h", 5.00000000e+01
    'Femtet.AddNewVariable "p", 5.00000000e-01

End Sub

'////////////////////////////////////////////////////////////
'    モデル作成関数
'////////////////////////////////////////////////////////////
Sub MakeModel()

    '------- Body配列変数の定義 -------
    Dim Body() as CGaudiBody

    '------- モデルを描画させない設定 -------
    Femtet.RedrawMode = False


    '------- SetPlane -------
    Dim Plane0 As new CGaudiPlane
    Plane0.Location.SetCoord 0.0, 0.0, 0.0
    Plane0.MainDirection.SetCoord 0.0, 1.0, 0.0
    Plane0.RefDirection.SetCoord 0.0, 0.0, 1.0
    Gaudi.SetPlane Plane0

    '------- CreateCylinder -------
    ReDim Preserve Body(0)
    Dim Point0 As new CGaudiPoint
    Point0.SetCoord 0, 0, 0
    Set Body(0) = Gaudi.CreateCylinder(Point0, 100, 300)

    '------- CreateCylinder -------
    ReDim Preserve Body(1)
    Dim Point1 As new CGaudiPoint
    Point1.SetCoord 0, 150-h/2, 0
    Set Body(1) = Gaudi.CreateCylinder(Point1, r, h)

    '------- SetName -------
    Body(0).SetName "Air", "000_空気"

    '------- SetName -------
    Body(1).SetName "固体", "固体"

    '------- AddBoundary -------
    Dim Face0 As CGaudiFace
    Set Face0 = Body(0).GetFaceByID(9)
    Face0.AddBoundary "in"

    '------- AddBoundary -------
    Dim Face1 As CGaudiFace
    Set Face1 = Body(0).GetFaceByID(15)
    Face1.AddBoundary "out"


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
    Gogh.Pascal.Vector = PASCAL_VELOCITY_C

    '------- 最大値の取得 -------
    Dim PosMax() As Double '最大値の座標
    Dim ResultMax As Double ' 最大値

    If Gogh.Pascal.GetMAXVectorPoint(VEC_C, CMPX_REAL_C, PosMax, ResultMax) = False Then
        Femtet.ShowLastError
    End If

    '------- 最小値の取得 -------
    Dim PosMin() As Double '最小値の座標
    Dim ResultMin As Double '最小値

    If Gogh.Pascal.GetMINVectorPoint(VEC_C, CMPX_REAL_C, PosMin, ResultMin) = False Then
        Femtet.ShowLastError
    End If

    '------- 任意座標の計算結果の取得 -------
    Dim Value() As New CComplex

    If Gogh.Pascal.GetVectorAtPoint(0, 0, 0, Value()) = False Then
        Femtet.ShowLastError
    End If

    ' 複数の座標の結果をまとめて取得する場合は、MultiGetVectorAtPoint関数をご利用ください。

End Sub

