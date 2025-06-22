# test_mail.py
from mail_config import EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER
import yagmail

yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
yag.send(
    to=EMAIL_RECEIVER,
    subject="Test d'envoi d'alerte",
    contents="Ceci est un test de l'envoi automatique d'e-mail depuis Python."
)
print("✅ E-mail envoyé avec succès.")
