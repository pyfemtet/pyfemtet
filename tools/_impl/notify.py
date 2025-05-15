import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from load_secret import secret
except ModuleNotFoundError:
    from .load_secret import secret


__all__ = ['send_mail']


# GmailのSMTPサーバー情報
smtp_server = secret['SMTP_SERVER']
smtp_port = secret['SMTP_PORT']

# 送信者のメールアドレスとパスワード
sender_email = secret['NOTIFICATION_FROM']

def send_mail(subject: str, body: str):

    body = body.replace('\\n', '\n')

    print()
    print('<subject>')
    try:
        print(subject)
    except UnicodeEncodeError:
        print(subject.encode('cp932', 'ignore').decode('cp932'))
    print()
    print('<body>')
    try:
        print(body)
    except UnicodeEncodeError:
        print(body.encode('cp932', 'ignore').decode('cp932'))
    print()


    # 受信者のメールアドレス
    receiver_email = ';'.join(secret['NOTIFICATION_TO'])

    # メールの作成
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # 本文をメールに添付
    msg.attach(MIMEText(body, 'plain'))

    # SMTPサーバーに接続（暗号化なし）
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        # ログインが不要な場合はloginしなくてOK
        server.sendmail(sender_email, receiver_email, msg.as_string())


if __name__ == '__main__':
    send_mail(sys.argv[1], sys.argv[2])
