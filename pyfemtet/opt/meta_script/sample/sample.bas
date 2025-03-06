Attribute VB_Name = "FemtetMacro"
Option Explicit

Dim Femtet As New CFemtet
Dim Als As CAnalysis
Dim BodyAttr As CBodyAttribute
Dim Bnd As CBoundary
Dim Mtl As CMaterial
Dim Gaudi As CGaudi
Dim Gogh As CGogh
'/////////////////////////null***�̐���/////////////////////////
'//���L�̎l�̕ϐ���CGaudi�N���XMulti***���g�p����ꍇ�ɗp���܂��B
'//�Ⴆ��MultiFillet���g�p����ꍇ�Ɉ����ł���Vertex(�_)�͎w�肹��
'//������Edge(��)������Fillet����ꍇ��nullVertex()��p���܂��B
'//�uGaudi.MultiFillet nullVertex,Edge�v�Ƃ����
'//������Edge����Fillet���邱�Ƃ��ł��܂��B
'/////////////////////////////////////////////////////////////
Global nullVertex() As CGaudiVertex
Global nullEdge() As CGaudiEdge
Global nullFace() As CGaudiFace
Global nullBody() As CGaudiBody

'///////////////////////////////////////////////////

'�ϐ��̐錾
Private pi as Double
Private c_pi as Double
Private section_radius as Double
Private coil_radius as Double
Private helical_pitch as Double
Private spiral_pitch as Double
Private n_turns as Double
'///////////////////////////////////////////////////


'////////////////////////////////////////////////////////////
'    Main�֐�
'////////////////////////////////////////////////////////////
Sub FemtetMain() 
    '------- Femtet�����N�� (�s�v�ȏꍇ��Excel�Ŏ��s���Ȃ��ꍇ�͉��s���R�����g�A�E�g���Ă�������) -------
    Workbooks("FemtetRef.xla").AutoExecuteFemtet

    '------- �V�K�v���W�F�N�g -------
    If Femtet.OpenNewProject() = False Then
        Femtet.ShowLastError
    End If

    '------- �ϐ��̒�` -------
    InitVariables

    '------- �f�[�^�x�[�X�̐ݒ� -------
    AnalysisSetUp
    BodyAttributeSetUp
    MaterialSetUp
    BoundarySetUp

    '------- ���f���̍쐬 -------
    Set Gaudi = Femtet.Gaudi
    MakeModel

    '------- �W�����b�V���T�C�Y�̐ݒ� -------
    '<<<<<<< �����v�Z�ɐݒ肷��ꍇ��-1��ݒ肵�Ă������� >>>>>>>
    Gaudi.MeshSize = 2.0

    '------- �v���W�F�N�g�̕ۑ� -------
    Dim ProjectFilePath As String
    ProjectFilePath = "E:\pyfemtet\pyfemtet\pyfemtet\opt\meta_script\sample\sample"
    '<<<<<<< �v���W�F�N�g��ۑ�����ꍇ�͈ȉ��̃R�����g���O���Ă������� >>>>>>>
    'If Femtet.SaveProject(ProjectFilePath & ".femprj", True) = False Then
    '    Femtet.ShowLastError
    'End If

    '------- ���b�V���̐��� -------
    '<<<<<<< ���b�V���𐶐�����ꍇ�͈ȉ��̃R�����g���O���Ă������� >>>>>>>
    'Gaudi.Mesh

    '------- ��͂̎��s -------
    '<<<<<<< ��͂����s����ꍇ�͈ȉ��̃R�����g���O���Ă������� >>>>>>>
    'Femtet.Solve

    '------- ��͌��ʂ̒��o -------
    '<<<<<<< �v�Z���ʂ𒊏o����ꍇ�͈ȉ��̃R�����g���O���Ă������� >>>>>>>
    'SamplingResult

    '------- �v�Z���ʂ̕ۑ� -------
    '<<<<<<< �v�Z����(.pdt)�t�@�C����ۑ�����ꍇ�͈ȉ��̃R�����g���O���Ă������� >>>>>>>
    'If Femtet.SavePDT(Femtet.ResultFilePath & ".pdt", True) = False Then
    '    Femtet.ShowLastError
    'End If

End Sub

'////////////////////////////////////////////////////////////
'    ��͏����̐ݒ�
'////////////////////////////////////////////////////////////
Sub AnalysisSetUp()

    '------- �ϐ��ɃI�u�W�F�N�g�̐ݒ� -------
    Set Als = Femtet.Analysis

    '------- ��͏�������(Common) -------
    Als.AnalysisType = GAUSS_C

    '------- ����(Gauss) -------
    Als.Gauss.b2ndEdgeElement = True

    '------- �d���g(Hertz) -------
    Als.Hertz.b2ndEdgeElement = True

    '------- ���d(Rayleigh) -------
    Als.Rayleigh.bConstantTemp = True

    '------- �J�����E(Open) -------
    Als.Open.OpenMethod = ABC_C
    Als.Open.ABCOrder = ABC_2ND_C

    '------- ���a���(Harmonic) -------
    Als.Harmonic.FreqSweepType = LINEAR_INTERVAL_C

    '------- ���g�H(WaveGuide) -------
    Als.WaveGuide.Result = FREQUENCY_C

    '------- ���x�Ȑݒ�(HighLevel) -------
    Als.HighLevel.nNonL = (20)
    Als.HighLevel.bATS = False
    Als.HighLevel.FactorType = RADIO_ANALYTICAL_C
    Als.HighLevel.bUseDeathMaterial = False

    '------- ���b�V���̐ݒ�(MeshProperty) -------
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

    '------- �����X�e�b�v�ݒ�(StepAnalysis) -------
    Als.StepAnalysis.bSetTime = True
    Als.StepAnalysis.Set_Table_withTime 0, (1.0), (20), (0.0)
    Als.StepAnalysis.BreakStep = (100)

    '------- ���ꍂ�����(LuvensSteadyState) -------
    Als.LuvensSteadyState.nEocn = (1000)

    '------- ����(Bernoulli) -------
    Als.Bernoulli.TurbulentModel = STANDARD_K_EPSILON_C
End Sub

'////////////////////////////////////////////////////////////
'    Body�����S�̂̐ݒ�
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp()

    '------- �ϐ��ɃI�u�W�F�N�g�̐ݒ� -------
    Set BodyAttr = Femtet.BodyAttribute

    '------- Body�����̐ݒ� -------
    BodyAttributeSetUp_Coil

    '+++++++++++++++++++++++++++++++++++++++++
    '++�g�p����Ă��Ȃ�BodyAttribute�f�[�^�ł�
    '++�g�p����ۂ̓R�����g���O���ĉ�����
    '+++++++++++++++++++++++++++++++++++++++++
    'BodyAttributeSetUp_Fe
    'BodyAttributeSetUp_Air
    'BodyAttributeSetUp_Air_Auto
End Sub

'////////////////////////////////////////////////////////////
'    Body�����̐ݒ� Body�������FCoil
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Coil()
    '------- Body������Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Body�����̒ǉ� -------
    BodyAttr.Add "Coil" 

    '------- Body���� Index�̐ݒ� -------
    Index = BodyAttr.Ask ( "Coil" ) 

    '------- �V�[�g�{�f�B�̌��� or 2������͂̉��s��(BodyThickness)/���C���[�{�f�B��(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False
    BodyAttr.Length(Index).BodyThickness = (1.0) * 10 ^ (-3)

    '------- �d��(Current) -------
    BodyAttr.Current(Index).I = (1)

    '------- �g�`(Waveform) -------
    BodyAttr.Current(Index).CurrentDirType = COIL_NORMAL_INOUTFLOW_INTERNAL_C
    BodyAttr.Current(Index).InFaceBodyKey = 0
    BodyAttr.Current(Index).InFaceTopolID = 26
    BodyAttr.Current(Index).OutFaceBodyKey = 0
    BodyAttr.Current(Index).OutFaceTopolID = 37

    '------- �������x(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- �d��(Electrode) -------
    BodyAttr.Electrode(Index).ElectricCondition = VOLT_C

    '------- �t��(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Body�����̐ݒ� Body�������FFe
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Fe()
    '------- Body������Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Body�����̒ǉ� -------
    BodyAttr.Add "Fe" 

    '------- Body���� Index�̐ݒ� -------
    Index = BodyAttr.Ask ( "Fe" ) 

    '------- �V�[�g�{�f�B�̌��� or 2������͂̉��s��(BodyThickness)/���C���[�{�f�B��(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False
    BodyAttr.Length(Index).BodyThickness = (1.0) * 10 ^ (-3)

    '------- �������x(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- �d��(Electrode) -------
    BodyAttr.Electrode(Index).ElectricCondition = VOLT_C

    '------- �t��(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Body�����̐ݒ� Body�������FAir
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Air()
    '------- Body������Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Body�����̒ǉ� -------
    BodyAttr.Add "Air" 

    '------- Body���� Index�̐ݒ� -------
    Index = BodyAttr.Ask ( "Air" ) 

    '------- �V�[�g�{�f�B�̌��� or 2������͂̉��s��(BodyThickness)/���C���[�{�f�B��(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = False
    BodyAttr.Length(Index).BodyThickness = (1.0) * 10 ^ (-3)

    '------- �������x(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = False

    '------- �d��(Electrode) -------
    BodyAttr.Electrode(Index).ElectricCondition = VOLT_C

    '------- ����(FluidBern) -------
    BodyAttr.FluidAttribute(Index).FlowCondition.bSpline = True

    '------- �t��(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Body�����̐ݒ� Body�������FAir_Auto
'////////////////////////////////////////////////////////////
Sub BodyAttributeSetUp_Air_Auto()
    '------- Body������Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Body�����̒ǉ� -------
    BodyAttr.Add "Air_Auto" 

    '------- Body���� Index�̐ݒ� -------
    Index = BodyAttr.Ask ( "Air_Auto" ) 

    '------- �V�[�g�{�f�B�̌��� or 2������͂̉��s��(BodyThickness)/���C���[�{�f�B��(WireWidth) -------
    BodyAttr.Length(Index).bUseAnalysisThickness2D = True

    '------- ��͗̈�(ActiveSolver) -------
    BodyAttr.ActiveSolver(Index).bWatt = False
    BodyAttr.ActiveSolver(Index).bGalileo = False

    '------- �������x(InitialVelocity) -------
    BodyAttr.InitialVelocity(Index).bAnalysisUse = True

    '------- �X�e�[�^/���[�^(StatorRotor) -------
    BodyAttr.StatorRotor(Index).State = AIR_C

    '------- �t��(Emittivity) -------
    BodyAttr.ThermalSurface(Index).Emittivity.Eps = (0.8)
End Sub

'////////////////////////////////////////////////////////////
'    Material�S�̂̐ݒ�
'////////////////////////////////////////////////////////////
Sub MaterialSetUp()

    '------- �ϐ��ɃI�u�W�F�N�g�̐ݒ� -------
    Set Mtl = Femtet.Material

    '------- Material�̐ݒ� -------
    MaterialSetUp_008_Copper

    '+++++++++++++++++++++++++++++++++++++++++
    '++�g�p����Ă��Ȃ�Material�f�[�^�ł�
    '++�g�p����ۂ̓R�����g���O���ĉ�����
    '+++++++++++++++++++++++++++++++++++++++++
    'MaterialSetUp_Air_Auto
End Sub

'////////////////////////////////////////////////////////////
'    Material�̐ݒ� Material���F008_Copper
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_008_Copper()
    '------- Material��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Material�̒ǉ� -------
    Mtl.Add "008_Copper" 

    '------- Material Index�̐ݒ� -------
    Index = Mtl.Ask ( "008_Copper" ) 

    '------- ������(Permeability) -------
    Mtl.Permeability(Index).BHExtrapolationType = BH_GRADIENT_LASTTWOPOINT_C

    '------- ��R��(Resistivity) -------
    Mtl.Resistivity(Index).sRho = (1.673) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_mRho 0, (1.0)
    Mtl.Resistivity(Index).Set_mRho 2, (1.0)
    Mtl.Resistivity(Index).Set_mRho 5, (1.0)
    Mtl.Resistivity(Index).Set_Table 0, (-195), (0.2) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 1, (0), (1.55) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 2, (100), (2.23) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 3, (300), (3.6) * 10 ^ (-8)
    Mtl.Resistivity(Index).Set_Table 4, (700), (6.7) * 10 ^ (-8)

    '------- ���d��(ElectricConductivity) -------
    Mtl.ElectricConductivity(Index).ConductorType = CONDUCTOR_C
    Mtl.ElectricConductivity(Index).sSigma = (5.977) * 10 ^ (7)

    '------- ��M(SpecificHeat) -------
    Mtl.SpecificHeat(Index).C = (0.385075378466) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 0, (-173), (0.2533597708746) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 1, (-73), (0.3575362729361) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 2, (20), (0.385075378466) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 3, (127), (0.3997104459761) * 10 ^ (3)
    Mtl.SpecificHeat(Index).Set_Table 4, (327), (0.4217417304) * 10 ^ (3)

    '------- ���x(Density) -------
    Mtl.Density(Index).Dens = (8.96) * 10 ^ (3)

    '------- �M�`����(ThermalConductivity) -------
    Mtl.ThermalConductivity(Index).sRmd = (401.9999999999999)
    Mtl.ThermalConductivity(Index).Set_Table 0, (-173), (482)
    Mtl.ThermalConductivity(Index).Set_Table 1, (-73), (415.0000000000001)
    Mtl.ThermalConductivity(Index).Set_Table 2, (27), (401.9999999999999)
    Mtl.ThermalConductivity(Index).Set_Table 3, (127), (393)
    Mtl.ThermalConductivity(Index).Set_Table 4, (227), (387)

    '------- ���c���W��(Expansion) -------
    Mtl.Expansion(Index).sAlf = (1.65) * 10 ^ (-5)
    Mtl.Expansion(Index).Set_vAlf 0, (0.0)
    Mtl.Expansion(Index).Set_vAlf 1, (0.0)
    Mtl.Expansion(Index).Set_vAlf 2, (0.0)

    '------- �e���萔(Elasticity) -------
    Mtl.Elasticity(Index).sY = (123) * 10 ^ (9)
    Mtl.Elasticity(Index).Nu = (0.35)

    '------- ���d�萔(PiezoElectricity) -------
    Mtl.PiezoElectricity(Index).bPiezo = False

    '------- ����(SoundVelocity) -------
    Mtl.SoundVelocity(Index).Vel0 = (331.0)

    '------- �S�x(Viscosity) -------
    Mtl.Viscosity(Index).Mu = (1.002) * 10 ^ (-3)

    '------- ����(Magnetize) -------
    Mtl.Magnetize(Index).MagRatioType = MAGRATIO_BR_C

    '------- ���� -------
    Mtl.Comment(Index).Comment = "�s�o�T�t " & Chr(13) & Chr(10) & "�@���x�F [C2]P.26" & Chr(13) & Chr(10) & "�@�e���萔�F [C2]P.26" & Chr(13) & Chr(10) & "�@���c���W���F [S]P.484" & Chr(13) & Chr(10) & "�@�������F [C2]P.507" & Chr(13) & Chr(10) & "�@��R���F [C2]P.490, [S]P.527" & Chr(13) & Chr(10) & "�@��M�F [S]P.473" & Chr(13) & Chr(10) & "�@�M�`�����F [C2]P.70" & Chr(13) & Chr(10) & "�@�@" & Chr(13) & Chr(10) & "�s�Q�l�����t�@" & Chr(13) & Chr(10) & "�@[C2] ���w�֗� ��b��II ����4�� ���{���w�w��� �ۑP(1993)" & Chr(13) & Chr(10) & "�@[S] ���ȔN�\ ����8�N �����V����� �ۑP(1996)"
End Sub

'////////////////////////////////////////////////////////////
'    Material�̐ݒ� Material���FAir_Auto
'////////////////////////////////////////////////////////////
Sub MaterialSetUp_Air_Auto()
    '------- Material��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Material�̒ǉ� -------
    Mtl.Add "Air_Auto" 

    '------- Material Index�̐ݒ� -------
    Index = Mtl.Ask ( "Air_Auto" ) 

    '------- �U�d��(Permittivity) -------
    Mtl.Permittivity(Index).sEps = (1.000517)

    '------- �ő�/����(SolidFluid) -------
    Mtl.SolidFluid(Index).State = FLUID_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�S�̂̐ݒ�
'////////////////////////////////////////////////////////////
Sub BoundarySetUp()

    '------- �ϐ��ɃI�u�W�F�N�g�̐ݒ� -------
    Set Bnd = Femtet.Boundary

    '------- Boundary�̐ݒ� -------
    BoundarySetUp_RESERVED_default

    '+++++++++++++++++++++++++++++++++++++++++
    '++�g�p����Ă��Ȃ�Boundary�f�[�^�ł�
    '++�g�p����ۂ̓R�����g���O���ĉ�����
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
'    Boundary�̐ݒ� Boundary���FRESERVED_default (�O�����E����)
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_RESERVED_default()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "RESERVED_default" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "RESERVED_default" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = ELECTRIC_WALL_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).Con = (2.33)
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(Fluid) -------
    Bnd.Fluid(Index).VelocityCondition = XYZ_VELOCITY_C

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FPort1
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Port1()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Port1" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Port1" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).Con = (2.33)
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(Fluid) -------
    Bnd.Fluid(Index).VelocityCondition = XYZ_VELOCITY_C

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FPort2
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Port2()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Port2" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Port2" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).Con = (2.33)
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C
    Bnd.Thermal(Index).RoomTemp.Temp = (0.0)

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(Fluid) -------
    Bnd.Fluid(Index).VelocityCondition = XYZ_VELOCITY_C

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FCoil_AutoPortIn
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_AutoPortIn()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Coil_AutoPortIn" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Coil_AutoPortIn" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FCoil_AutoPortOut
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_AutoPortOut()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Coil_AutoPortOut" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Coil_AutoPortOut" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FCoil_PortInAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_PortInAuto()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Coil_PortInAuto" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Coil_PortInAuto" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FCoil_PortOutAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_PortOutAuto()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Coil_PortOutAuto" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Coil_PortOutAuto" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FCoil_InAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_InAuto()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Coil_InAuto" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Coil_InAuto" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    Boundary�̐ݒ� Boundary���FCoil_OutAuto
'////////////////////////////////////////////////////////////
Sub BoundarySetUp_Coil_OutAuto()
    '------- Boundary��Index��ۑ�����ϐ� -------
    Dim Index As Integer

    '------- Boundary�̒ǉ� -------
    Bnd.Add "Coil_OutAuto" 

    '------- Boundary Index�̐ݒ� -------
    Index = Bnd.Ask ( "Coil_OutAuto" ) 

    '------- �d�C(Electrical) -------
    Bnd.Electrical(Index).Condition = PORT_C

    '------- �M(Thermal) -------
    Bnd.Thermal(Index).bConAuto = False
    Bnd.Thermal(Index).bSetRadioSetting = False

    '------- ����_�����x(RoomTemp) -------
    Bnd.Thermal(Index).RoomTemp.TempType = TEMP_AMBIENT_C

    '------- �t��(Emittivity) -------
    Bnd.Thermal(Index).Emittivity.Eps = (0.999)

    '------- ����(FluidBern) -------
    Bnd.FluidBern(Index).P = (0.0)

    '------- ���z(Distribution) -------
    Bnd.FluidBern(Index).TempType = TEMP_DIRECT_C
End Sub

'////////////////////////////////////////////////////////////
'    IF�֐�
'////////////////////////////////////////////////////////////
Function F_IF(expression As Double, val_true As Double, val_false As Double) As Double
    If expression Then
        F_IF = val_true
    Else
        F_IF = val_false
    End If

End Function

'////////////////////////////////////////////////////////////
'    �ϐ���`�֐�
'////////////////////////////////////////////////////////////
Sub InitVariables()


    'VB��̕ϐ��̒�`
    pi = 3.1415926535897932
    c_pi = 3.1415926535897932
    section_radius = 2
    coil_radius = 10
    helical_pitch = 6
    spiral_pitch = 0
    n_turns = 5

    'FemtetGUI��̕ϐ��̓o�^�i�������f���̕ϐ����䓙�ł̂ݗ��p�j
    'Femtet.AddNewVariable "c_pi", 3.14159265e+00
    'Femtet.AddNewVariable "section_radius", 2.00000000e+00
    'Femtet.AddNewVariable "coil_radius", 1.00000000e+01
    'Femtet.AddNewVariable "helical_pitch", 6.00000000e+00
    'Femtet.AddNewVariable "spiral_pitch", 0.00000000e+00
    'Femtet.AddNewVariable "n_turns", 5.00000000e+00

End Sub

'////////////////////////////////////////////////////////////
'    ���f���쐬�֐�
'////////////////////////////////////////////////////////////
Sub MakeModel()

    '------- Body�z��ϐ��̒�` -------
    Dim Body() as CGaudiBody

    '------- ���f����`�悳���Ȃ��ݒ� -------
    Femtet.RedrawMode = False


    '------- CreateHelicalCylinder -------
    ReDim Preserve Body(0)
    Dim Point0 As new CGaudiPoint
    Dim Point1 As new CGaudiPoint
    Point0.SetCoord 0, 0, 0
    Point1.SetCoord coil_radius, 0, 0
    Set Body(0) = Gaudi.CreateHelicalCylinder(Point0, section_radius, Point1, helical_pitch, spiral_pitch, n_turns, True)

    '------- SetName -------
    Body(0).SetName "Coil", "008_��Cu"

    '------- SetName -------
    Body(0).SetName "", "008_Copper"


    '------- ���f�����ĕ`�悵�܂� -------
    Femtet.Redraw

End Sub

'////////////////////////////////////////////////////////////
'    �v�Z���ʒ��o�֐�
'////////////////////////////////////////////////////////////
Sub SamplingResult()

    '------- �ϐ��ɃI�u�W�F�N�g�̐ݒ� -------
    Set Gogh = Femtet.Gogh

    '------- ���݂̌v�Z���ʂ𒆊ԃt�@�C������J�� -------
    If Femtet.OpenCurrentResult(True) = False Then
        Femtet.ShowLastError
    End If

    '------- �t�B�[���h�̐ݒ� -------
    Gogh.Gauss.Vector = GAUSS_MAGNETIC_FLUX_DENSITY_C

    '------- �ő�l�̎擾 -------
    Dim PosMax() As Double '�ő�l�̍��W
    Dim ResultMax As Double ' �ő�l

    If Gogh.Gauss.GetMAXVectorPoint(VEC_C, CMPX_REAL_C, PosMax, ResultMax) = False Then
        Femtet.ShowLastError
    End If

    '------- �ŏ��l�̎擾 -------
    Dim PosMin() As Double '�ŏ��l�̍��W
    Dim ResultMin As Double '�ŏ��l

    If Gogh.Gauss.GetMINVectorPoint(VEC_C, CMPX_REAL_C, PosMin, ResultMin) = False Then
        Femtet.ShowLastError
    End If

    '------- �C�Ӎ��W�̌v�Z���ʂ̎擾 -------
    Dim Value() As New CComplex

    If Gogh.Gauss.GetVectorAtPoint(0, 0, 0, Value()) = False Then
        Femtet.ShowLastError
    End If

    ' �����̍��W�̌��ʂ��܂Ƃ߂Ď擾����ꍇ�́AMultiGetVectorAtPoint�֐��������p���������B

End Sub

