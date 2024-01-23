import os
import sys
import json
import NXOpen


def main(prtPath:str, parameters:'dict as str', x_tPath:str = None):
    '''
    .prt ファイルのパスを受け取り、parameters に指定された変数を更新し、
    x_tPath が None のときは.prt と同じディレクトリに
    .x_t ファイルをエクスポートする

    Parameters
    ----------
    prtPath : str
        DESCRIPTION.
    parameters : 'dict as str'
        DESCRIPTION.

    Returns
    -------
    None.

    '''    
    # 保存先の設定
    prtPath = os.path.abspath(prtPath) # 一応
    if x_tPath is None:
        x_tPath = os.path.splitext(prtPath)[0] + '.x_t'

    # 辞書の作成
    parameters = json.loads(parameters)
    
    # session の取得とパートを開く
    theSession = NXOpen.Session.GetSession()
    theSession.Parts.OpenActiveDisplay(prtPath, NXOpen.DisplayPartOption.AllowAdditional)
    theSession.ApplicationSwitchImmediate("UG_APP_MODELING")

    # part の設定
    workPart = theSession.Parts.Work
    displayPart = theSession.Parts.Display

    # 式を更新
    unit_mm = workPart.UnitCollection.FindObject("MilliMeter")
    for k, v in parameters.items():
        try:
            exp = workPart.Expressions.FindObject(k)
        except NXOpen.NXException:
            print(f'├ 変数{k}は .prt ファイルに含まれていません。無視されます。')
            continue

        workPart.Expressions.EditWithUnits(exp, unit_mm, str(v))
        # 式の更新を適用
        id1 = theSession.NewestVisibleUndoMark
        try:
            nErrs1 = theSession.UpdateManager.DoUpdate(id1)
        # 更新に失敗
        except NXOpen.NXException as e:
            print('└ 形状が破綻しました。操作を取り消します。')
            return None

    print('│ model 更新に成功しました。')

    try:
        # parasolid のエクスポート
        parasolidExporter1 = theSession.DexManager.CreateParasolidExporter()

        parasolidExporter1.ObjectTypes.Curves = False
        parasolidExporter1.ObjectTypes.Surfaces = False
        parasolidExporter1.ObjectTypes.Solids = True

        parasolidExporter1.InputFile = prtPath
        parasolidExporter1.ParasolidVersion = NXOpen.ParasolidExporter.ParasolidVersionOption.Current
        parasolidExporter1.OutputFile = x_tPath

        parasolidExporter1.Commit()

        parasolidExporter1.Destroy()
    except:
        print('└ parasolid 更新に失敗しました。')
        return None

    print('└ parasolid 更新が正常に終了しました。')
    return None



if __name__ == "__main__":
    print('---NX started---')
    print('current directory: ', os.getcwd())
    print('arguments: ')
    for arg in sys.argv[1:]:
        print('│ ', arg)
    main(*sys.argv[1:])
    print('---NX end---')
