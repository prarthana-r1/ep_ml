import dotenv from 'dotenv'
dotenv.config();   // 👈 load first!
import pkg from 'pg';
const { Pool } = pkg;


console.log("DATABASE_URL:", process.env.DATABASE_URL);
const pool = new Pool({ connectionString: process.env.DATABASE_URL });


export async function query(sql, params) {
const client = await pool.connect();
try {
const res = await client.query(sql, params);
return res;
} finally {
client.release();
}
}


export default { query };