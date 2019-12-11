# send email from gmail account  
def send_email(subject, email_body, sender_email_id_password, sender_email_id="xxxxxxxxx@gmail.com", receiver_email_id="xxxxxxxx@yahoo.com"):
	# import smtplib
	import smtplib
	# creates SMTP session 
	mail = smtplib.SMTP('smtp.gmail.com', 587) 

	mail.ehlo()  
	# start TLS for security 
	mail.starttls()
	# call again .ehlo()
	mail.ehlo() 
	  
	# Authentication 
	mail.login(sender_email_id, sender_email_id_password) 
	
	# combine email subject and email body
	message = 'Subject: {}\n\n{}'.format(subject, email_body)	
	
	# sending the mail 
	mail.sendmail(sender_email_id, receiver_email_id, message) 
	  
	# terminating the session 
	mail.quit()



    


    




