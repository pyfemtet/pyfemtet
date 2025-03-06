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
Private section_radius as Double
Private coil_radius as Double
Private helical_pitch as Double
Private spiral_pitch as Double
Private n_turns as Double
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
    Gaudi.MeshSize = 2.0

    '------- プロジェクトの保存 -------
    Dim ProjectFilePath As String
    ProjectFilePath = "E:\pyfemtet\pyfemtet\pyfemtet\opt\meta_script\sample\sample"
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
    Als.AnalysisType = GAUSS_C

    '------- 磁場(Gauss) -------
    Als.Gauss.b2ndEdgeElement = True

    '------- 電磁波(Hertz) -------
    Als.Hertz.b2ndEdgeElement = True

    '------- 圧電(Rayleigh) -------
    Als.Rayleigh.bConstantTemp = True

    '------- 開放境界(Open) -------
    Als.Open.OpenMethod = ABC_C
    Als.Open.ABCOrder = ABC_2ND_C

    '------- 調和解析(Harmonic) -------
    Als.Harmonic.FreqSweepType = LINEAR_INTERVAL_C

    '------- 導波路(WaveGuide) -------
    Als.WaveGuide.Result = FREQUENCY_C

    '------- 高度な設定(HighLevel) -------
    Als.HighLevel.nNonL = (20)
    Als.HighLevel.bATS = False
    Als.HighLevel.FactorType = RADIO_ANALYTICAL_C
    Als.HighLevel.bUseDeathMaterial = False

    '------- メッシュの設定(MeshProperty) -------
    Als.MeshProperty.nMGElementIncRate = (50)
    Als.MeshProperty.nMG = (100)
    Als.MeshProperty.nMaxAdaptive = (10)
    Als.MeshProperty.nMaxAdaptivePort = (15)
    Als.MeshProperty.AdaptiveTolPort = (1.0) * 10 ^ (-2)
    Als.MeshProperty.AdaptiveEnergyType = IND_C
    Als.MeshProperty.bAutoAir = True
    Als.MeshProperty.AutoAirMeshSize = (22.77826356331989)
    Als.MeshProperty.bChangePlane = True
    Als.MeshProperty.bMeshG2 = True
    Als.MeshProperty.bPeriodMesh = False

    '------- 複数ステップ設定(StepAnalysis) -------
    Als.StepAnalysis.bSetTime = True
    Als.StepAnalysis.Set_Table_withTime 0, (1.0), (20), (0.0)
    Als.StepAnalysis.BreakStep = (100)

    '------- 定常場高速解析(LuvensSteadyState) -------
    Als.LuvensSteadyState.nEocn = (1000)

    '------- 流体(Bernoulli) -------
    Als.Bernoulli.TurbulentModel = STANDARD_K_EPSILON_C
End Sub

'////////////////////////////////////////////////////////////
'    Body属性全体の設定
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp()

    '------- 変数にオブジェクトの設定 -------
    Set BodyAttr = Femtet.BodyAttribute

    '------- Body属性の設定 -------
    BodyAttributeSetUp_Coil

    '+++++++++++++++++++++++++++++++++++++++++
    '++使用されていないBodyAttributeデータです
    '++使用する際はコメントを外して下さい
    '+++++++++++++++++++++++++++++++++++++++++
    'BodyAttributeSetUp_Fe
    'BodyAttributeSetUp_Air
    'BodyAttributeSetUp_Air_Auto
End Sub

'////////////////////////////////////////////////////////////
'    Body属性の設定 Body属性名：Coil
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Coil()
    '------- Body属性のIndexを保存する変数 -------
    Dim Index As Integer

    '------- Body属性の追加 -------
    BodyAttr.Add "Coil" 

    '------- Body属性 Indexの設定 -------
    Index = BodyAttr.Ask ( "Coil" ) 

    '------- シートボディの厚み or 2次元解析の奥行き(BodyThickness)/ワイヤーボディ幅(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False
    BodyAttr.Length(Index).BodyThickness = (1.0) * 10 ^ (-3)

    '------- 電流(Current) -------
    BodyAttr.Current(Index).I = (1)

    '------- 波形(Waveform) -------
    BodyAttr.Current(Index).CurrentDirType = COIL_NORMAL_INOUTFLOW_INTERNAL_C
    BodyAttr.Current(Index).InFaceBodyKey = 0
    BodyAttr.Current(Index).InFaceTopolID = 26
    BodyAttr.Current(Index).OutFaceBodyKey = 0
    BodyAttr.Current(Index).OutFaceTopolID = 37

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- 電極(Electrode) -------
    BodyAttr.Electrode(Index).ElectricCondition = VOLT_C

    '------- 輻射(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Body属性の設定 Body属性名：Fe
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Fe()
    '------- Body属性のIndexを保存する変数 -------
    Dim Index As Integer

    '------- Body属性の追加 -------
    BodyAttr.Add "Fe" 

    '------- Body属性 Indexの設定 -------
    Index = BodyAttr.Ask ( "Fe" ) 

    '------- シートボディの厚み or 2次元解析の奥行き(BodyThickness)/ワイヤーボディ幅(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False
    BodyAttr.Length(Index).BodyThickness = (1.0) * 10 ^ (-3)

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- 電極(Electrode) -------
    BodyAttr.Electrode(Index).ElectricCondition = VOLT_C

    '------- 輻射(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
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
    BodyAttr.Length(Index).BodyThickness = (1.0) * 10 ^ (-3)

    '------- 初期速度(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- 電極(Electrode) -------
    BodyAttr.Electrode(Index).ElectricCondition = VOLT_C

    '------- 流体(FluidBern) -------
    BodyAttr.FluidAttribute(Index).FlowCondition.bSpline = True

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
    MaterialSetUp_008_Copper

    '+++++++++++++++++++++++++++++++++++++++++
    '++使用されていないMaterialデータです
    '++使用する際はコメントを外して下さい
    '+++++++++++++++++++++++++++++++++++++++++
    'MaterialSetUp_Air_Auto
End Sub

'////////////////////////////////////////////////////////////
'    Materialの設定 Material名：008_Copper
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_008_Copper()
    '------- MaterialのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Materialの追加 -------
    Mtl.Add "008_Copper" 

    '------- Material Indexの設定 -------
    Index = Mtl.Ask ( "008_Copper" ) 

    '------- 透磁率(Permeability) -------
    Mtl.Permeability(Index).BHExtrapolationType = BH_GRADIENT_LASTTWOPOINT_C

    '------- 抵抗率(Resistivity) -------
    Mtl.Resistivity(Index).sRho = (1.673) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_mRho 0, (1.0)
    Mtl.Resistivity(Index).Set_mRho 2, (1.0)
    Mtl.Resistivity(Index).Set_mRho 5, (1.0)
    Mtl.Resistivity(Index).Set_Table 0, (-195), (0.2) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 1, (0), (1.55) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 2, (100), (2.23) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 3, (300), (3.6) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 4, (700), (6.7) * 10 ^ (-8)

    '------- 導電率(ElectricConductivity) -------
    Mtl.ElectricConductivity(Index).ConductorType = CONDUCTOR_C
    Mtl.ElectricConductivity(Index).sSigma = (5.977) * 10 ^ (7)

    '------- 比熱(SpecificHeat) -------
    Mtl.SpecificHeat(Index).C = (0.385075378466) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 0, (-173), (0.2533597708746) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 1, (-73), (0.3575362729361) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 2, (20), (0.385075378466) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 3, (127), (0.3997104459761) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 4, (327), (0.4217417304) * 10 ^ (3)

    '------- 密度(Density) -------
    Mtl.Density(Index).Dens = (8.96) * 10 ^ (3)

    '------- 熱伝導率(ThermalConductivity) -------
    Mtl.ThermalConductivity(Index).sRmd = (401.9999999999999)
    Mtl.ThermalConductivity(Index).Set_Table 0, (-173), (482)
    Mtl.ThermalConductivity(Index).Set_Table 1, (-73), (415.0000000000001)
    Mtl.ThermalConductivity(Index).Set_Table 2, (27), (401.9999999999999)
    Mtl.ThermalConductivity(Index).Set_Table 3, (127), (393)
    Mtl.ThermalConductivity(Index).Set_Table 4, (227), (387)

    '------- 線膨張係数(Expansion) -------
    Mtl.Expansion(Index).sAlf = (1.65) * 10 ^ (-5)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- 弾性定数(Elasticity) -------
    Mtl.Elasticity(Index).sY = (123) * 10 ^ (9)
    Mtl.Elasticity(Index).Nu = (0.35)

    '------- 圧電定数(PiezoElectricity) -------
    Mtl.PiezoElectricity(Index).bPiezo = False

    '------- 音速(SoundVelocity) -------
    Mtl.SoundVelocity(Index).Vel0 = (331.0)

    '------- 粘度(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- 着磁(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C

    '------- 説明 -------
    Mtl.Comment(Index).Comment = "《出典》 " & Chr(13) & Chr(10) & "　密度： [C2]P.26" & Chr(13) & Chr(10) & "　弾性定数： [C2]P.26" & Chr(13) & Chr(10) & "　線膨張係数： [S]P.484" & Chr(13) & Chr(10) & "　透磁率： [C2]P.507" & Chr(13) & Chr(10) & "　抵抗率： [C2]P.490, [S]P.527" & Chr(13) & Chr(10) & "　比熱： [S]P.473" & Chr(13) & Chr(10) & "　熱伝導率： [C2]P.70" & Chr(13) & Chr(10) & "　　" & Chr(13) & Chr(10) & "《参考文献》　" & Chr(13) & Chr(10) & "　[C2] 化学便覧 基礎編II 改訂4版 日本化学学会編 丸善(1993)" & Chr(13) & Chr(10) & "　[S] 理科年表 平成8年 国立天文台編 丸善(1996)"
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

    '+++++++++++++++++++++++++++++++++++++++++
    '++使用されていないBoundaryデータです
    '++使用する際はコメントを外して下さい
    '+++++++++++++++++++++++++++++++++++++++++
    'BoundarySetUp_Port1
    'BoundarySetUp_Port2
    'BoundarySetUp_Coil_AutoPortIn
    'BoundarySetUp_Coil_AutoPortOut
    'BoundarySetUp_Coil_PortInAuto
    'BoundarySetUp_Coil_PortOutAuto
    'BoundarySetUp_Coil_InAuto
    'BoundarySetUp_Coil_OutAuto
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

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = ELECTRIC_WALL_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).Con = (2.33)
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(Fluid) -------
    Bnd.Fluid(Index).VelocityCondition = XYZ_VELOCITY_C

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Port1
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Port1()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Port1" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Port1" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).Con = (2.33)
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(Fluid) -------
    Bnd.Fluid(Index).VelocityCondition = XYZ_VELOCITY_C

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Port2
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Port2()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Port2" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Port2" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).Con = (2.33)
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(Fluid) -------
    Bnd.Fluid(Index).VelocityCondition = XYZ_VELOCITY_C

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Coil_AutoPortIn
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_AutoPortIn()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Coil_AutoPortIn" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Coil_AutoPortIn" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Coil_AutoPortOut
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_AutoPortOut()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Coil_AutoPortOut" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Coil_AutoPortOut" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Coil_PortInAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_PortInAuto()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Coil_PortInAuto" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Coil_PortInAuto" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Coil_PortOutAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_PortOutAuto()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Coil_PortOutAuto" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Coil_PortOutAuto" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Coil_InAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_InAuto()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Coil_InAuto" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Coil_InAuto" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- 分布(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundaryの設定 Boundary名：Coil_OutAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_OutAuto()
    '------- BoundaryのIndexを保存する変数 -------
    Dim Index As Integer

    '------- Boundaryの追加 -------
    Bnd.Add "Coil_OutAuto" 

    '------- Boundary Indexの設定 -------
    Index = Bnd.Ask ( "Coil_OutAuto" ) 

    '------- 電気(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- 熱(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- 室温_環境温度(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- 輻射(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- 流体(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

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
    section_radius = 2
    coil_radius = 10
    helical_pitch = 6
    spiral_pitch = 0
    n_turns = 5

    'FemtetGUI上の変数の登録（既存モデルの変数制御等でのみ利用）
    'Femtet.AddNewVariable "c_pi", 3.14159265e+00
    'Femtet.AddNewVariable "section_radius", 2.00000000e+00
    'Femtet.AddNewVariable "coil_radius", 1.00000000e+01
    'Femtet.AddNewVariable "helical_pitch", 6.00000000e+00
    'Femtet.AddNewVariable "spiral_pitch", 0.00000000e+00
    'Femtet.AddNewVariable "n_turns", 5.00000000e+00

End Sub

'////////////////////////////////////////////////////////////
'    モデル作成関数
'////////////////////////////////////////////////////////////
Sub MakeModel()

    '------- Body配列変数の定義 -------
    Dim Body() as CGaudiBody

    '------- モデルを描画させない設定 -------
    Femtet.RedrawMode = False


    '------- CreateHelicalCylinder -------
    ReDim Preserve Body(0)
    Dim Point0 As new CGaudiPoint
    Dim Point1 As new CGaudiPoint
    Point0.SetCoord 0, 0, 0
    Point1.SetCoord coil_radius, 0, 0
    Set Body(0) = Gaudi.CreateHelicalCylinder(Point0, section_radius, Point1, helical_pitch, spiral_pitch, n_turns, True)

    '------- SetName -------
    Body(0).SetName "Coil", "008_銅Cu"

    '------- SetName -------
    Body(0).SetName "", "008_Copper"


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
    Gogh.Gauss.Vector = GAUSS_MAGNETIC_FLUX_DENSITY_C

    '------- 最大値の取得 -------
    Dim PosMax() As Double '最大値の座標
    Dim ResultMax As Double ' 最大値

    If Gogh.Gauss.GetMAXVectorPoint(VEC_C, CMPX_REAL_C, PosMax, ResultMax) = False Then
        Femtet.ShowLastError
    End If

    '------- 最小値の取得 -------
    Dim PosMin() As Double '最小値の座標
    Dim ResultMin As Double '最小値

    If Gogh.Gauss.GetMINVectorPoint(VEC_C, CMPX_REAL_C, PosMin, ResultMin) = False Then
        Femtet.ShowLastError
    End If

    '------- 任意座標の計算結果の取得 -------
    Dim Value() As New CComplex

    If Gogh.Gauss.GetVectorAtPoint(0, 0, 0, Value()) = False Then
        Femtet.ShowLastError
    End If

    ' 複数の座標の結果をまとめて取得する場合は、MultiGetVectorAtPoint関数をご利用ください。

End Sub

