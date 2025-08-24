import nodemailer from 'nodemailer';
 function transporter() {
 return nodemailer.createTransport({
 host: process.env.SMTP_HOST,
 port: Number(process.env.SMTP_PORT || 587),
 secure: false,
 auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
 });
 }
 export async function sendAlert({ to, subject, text }) {
 const t = transporter();
 return t.sendMail({ from: process.env.ALERT_EMAIL_FROM, to, subject, text });
 }