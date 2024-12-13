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
Private width as Double
Private depth as Double
Private height as Double
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
    ProjectFilePath = "C:\Users\mm11592\Documents\myFiles2\working\1_PyFemtetOpt\PyFemtetDev3\pyfemtet\tests\excel_ui\プロジェクト"
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
    Als.AnalysisType = COULOMB_C

    '------- 磁場(Gauss) -------
    Als.Gauss.b2ndEdgeElement = True

    '------- 電磁波(Hertz) -------
    Als.Hertz.b2ndEdgeElement = True

    '------- 開放境界(Open) -------
    Als.Open.OpenMethod = ABC_C

    '------- 調和解析(Harmonic) -------
    Als.Harmonic.FreqSweepType = LINEAR_INTERVAL_C

    '------- 高度な設定(HighLevel) -------
    Als.HighLevel.MemoryLimit = (16)

    '------- メッシュの設定(MeshProperty) -------
    Als.MeshProperty.bAdaptiveMeshOnCurve = True
    Als.MeshProperty.bAutoAir = True
    Als.MeshProperty.AutoAirMeshSize = (1.8)
    Als.MeshProperty.bChangePlane = True
    Als.MeshProperty.bMeshG2 = True
    Als.MeshProperty.bPeriodMesh = False

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

    '+++++++++++++++++++++++++++++++++++++++++
    '++使用されていないBodyAttributeデータです
    '++使用する際はコメントを外して下さい
    '+++++++++++++++++++++++++++++++++++++++++
    'BodyAttributeSetUp_Air_Auto
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
    BodyAttr.Direction(Index).SetAxisVector (0.0), (0.0), (1.0)

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = True

    '------- 輻射(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Body属性の設定 Body属性名：Air_Auto
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Air_Auto()
    '------- Body属性のIndexを保存する変数 -------
    Dim Index As Integer

    '------- Body属性の追加 -------
    BodyAttr.Add "Air_Auto" 

    '------- Body属性 Indexの設定 -------
    Index = BodyAttr.Ask ( "Air_Auto" ) 

    '------- シートボディの厚み or 2次元解析の奥行き(BodyThickness)/ワイヤーボディ幅(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = True

    '------- 解析領域(ActiveSolver) -------
    BodyAttr.ActiveSolver(Index).bWatt = False
    BodyAttr.ActiveSolver(Index).bGalileo = False

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = True

    '------- ステータ/ロータ(StatorRotor) -------
    BodyAttr.StatorRotor(Index).State = AIR_C

    '------- 流体(FluidBern) -------
    BodyAttr.FluidAttribute(Index).FlowCondition.bSpline = False

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
    MaterialSetUp_001_アルミナ

    '+++++++++++++++++++++++++++++++++++++++++
    '++使用されていないMaterialデータです
    '++使用する際はコメントを外して下さい
    '+++++++++++++++++++++++++++++++++++++++++
    'MaterialSetUp_Air_Auto
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

    '------- 誘電体(Permittivity) -------
    Mtl.Permittivity(Index).TanD = (0.002)
    Mtl.Permittivity(Index).sEps = (8.5)

    '------- 抵抗率(Resistivity) -------
    Mtl.Resistivity(Index).ResistivityUse = USE_DISABLE_C

    '------- 密度(Density) -------
    Mtl.Density(Index).Dens = (3800)

    '------- 熱伝導率(ThermalConductivity) -------
    Mtl.ThermalConductivity(Index).sRmd = (33)

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (5.4) * 10 ^ (-6)

    '------- 弾性定数(Elasticity) -------
    Mtl.Elasticity(Index).sY = (220) * 10 ^ (9)

    '------- 圧電定数(PiezoElectricity) -------
    Mtl.PiezoElectricity(Index).bPiezo = False

    '------- 説明 -------
    Mtl.Comment(Index).Comment = "《出典》 " & Chr(13) & Chr(10) & "　密度： [C1]P.590" & Chr(13) & Chr(10) & "　弾性定数： [C1]P.590" & Chr(13) & Chr(10) & "　線膨張係数： [C1]P.590" & Chr(13) & Chr(10) & "　誘電率： [S]P.530" & Chr(13) & Chr(10) & "　抵抗率： [S]P.534" & Chr(13) & Chr(10) & "　熱伝導率： [C1]P.590" & Chr(13) & Chr(10) & "　 " & Chr(13) & Chr(10) & "《参考文献》 " & Chr(13) & Chr(10) & "　[C1] 化学便覧 基礎編I 改訂4版 日本化学学会編 丸善(1993)" & Chr(13) & Chr(10) & "　[S] 理科年表 平成8年 国立天文台編 丸善(1996)" & Chr(13) & Chr(10) & "　 "
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：Air_Auto
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_Air_Auto()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "Air_Auto" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "Air_Auto" ) 

    '------- 誘電体(Permittivity) -------
    Mtl.Permittivity(Index).sEps = (1.000517)

    '------- 抵抗率(Resistivity) -------
    Mtl.Resistivity(Index).ResistivityUse = USE_DISABLE_C

    '------- 固体/流体(SolidFluid) -------
    Mtl.SolidFluid(Index).State = FLUID_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary全体の設定
'////////////////////////////////////////////////////////////
Sub BoundarySetUp()

    '------- 変数にオブジェクトの設定 -------
    Set Bnd = Femtet.Boundary

    '------- Boundaryの設定 -------
    BoundarySetUp_RESERVED_default
    BoundarySetUp_境界条件_001
    BoundarySetUp_境界条件_002
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

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).bSpline = False
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：境界条件_001
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_境界条件_001()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "境界条件_001" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "境界条件_001" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = ELECTRIC_WALL_C
    Bnd.Electrical(Index).V = (1)

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = True
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.8)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).bSpline = False
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：境界条件_002
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_境界条件_002()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "境界条件_002" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "境界条件_002" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = ELECTRIC_WALL_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = True
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.8)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).bSpline = False
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
    width = 3
    depth = 3
    height = 3

    'FemtetGUI上の変数の登録（既存モデルの変数制御等でのみ利用）
    'Femtet.AddNewVariable "width", 3.00000000e+00
    'Femtet.AddNewVariable "depth", 3.00000000e+00
    'Femtet.AddNewVariable "height", 3.00000000e+00

End Sub

'////////////////////////////////////////////////////////////
'    モデル作成関数
'////////////////////////////////////////////////////////////
Sub MakeModel()

    '------- Body配列変数の定義 -------
    Dim Body() as CGaudiBody

    '------- モデルを描画させない設定 -------
    Femtet.RedrawMode = False


    '------- CreateBox -------
    ReDim Preserve Body(0)
    Dim Point0 As new CGaudiPoint
    Point0.SetCoord 0, 0, 0
    Set Body(0) = Gaudi.CreateBox(Point0, width, depth, height)

    '------- SetName -------
    Body(0).SetName "ボディ属性_001", "001_アルミナ"

    '------- SetBoundary -------
    Dim Face0 As CGaudiFace
    Set Face0 = Body(0).GetFaceByID(36)
    Face0.SetBoundary "境界条件_001"

    '------- SetBoundary -------
    Dim Face1 As CGaudiFace
    Set Face1 = Body(0).GetFaceByID(9)
    Face1.SetBoundary "境界条件_002"


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
    Gogh.Coulomb.Potential = COULOMB_VOLTAGE_C

    '------- 最大値の取得 -------
    Dim PosMax() As Double '最大値の座標
    Dim ResultMax As Double ' 最大値

    If Gogh.Coulomb.GetMAXPotentialPoint(CMPX_REAL_C, PosMax, ResultMax) = False Then
        Femtet.ShowLastError
    End If

    '------- 最小値の取得 -------
    Dim PosMin() As Double '最小値の座標
    Dim ResultMin As Double '最小値

    If Gogh.Coulomb.GetMINPotentialPoint(CMPX_REAL_C, PosMin, ResultMin) = False Then
        Femtet.ShowLastError
    End If

    '------- 任意座標の計算結果の取得 -------
    Dim Value As New CComplex

    If Gogh.Coulomb.GetPotentialAtPoint(3, 3, 0, Value) = False Then
        Femtet.ShowLastError
    End If

    ' 複数の座標の結果をまとめて取得する場合は、MultiGetPotentialAtPoint関数をご利用ください。

End Sub

