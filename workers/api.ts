// CloudFlare Worker API for Lamination DeviceLink
// Handles auth (Google OAuth, JWT), model storage (R2), data collection

interface Env {
  DB: D1Database;
  MODELS: R2Bucket;
  DATA: R2Bucket;
  ENV: string;
  JWT_SECRET: string;
  GOOGLE_CLIENT_ID: string;
  GOOGLE_CLIENT_SECRET: string;
  ORIGIN: string; // e.g. https://lamination.pages.dev
}

// ---- JWT helpers ----
async function createJWT(payload: Record<string, any>, secret: string): Promise<string> {
  const header = { alg: 'HS256', typ: 'JWT' };
  const b64 = (o: any) => btoa(JSON.stringify(o)).replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_');
  const h = b64(header);
  const p = b64(payload);
  const sig = await crypto.subtle.sign(
    { name: 'HMAC', hash: 'SHA-256' },
    await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']),
    new TextEncoder().encode(`${h}.${p}`)
  );
  const s = b64(String.fromCharCode(...new Uint8Array(sig)));
  return `${h}.${p}.${s}`;
}

async function verifyJWT(token: string, secret: string): Promise<Record<string, any> | null> {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const sig = await crypto.subtle.sign(
      { name: 'HMAC', hash: 'SHA-256' },
      await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']),
      new TextEncoder().encode(`${parts[0]}.${parts[1]}`)
    );
    const expected = btoa(String.fromCharCode(...new Uint8Array(sig))).replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_');
    if (parts[2] !== expected) return null;
    return JSON.parse(atob(parts[1]));
  } catch { return null; }
}

// ---- Responses ----
function json(data: any, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type,Authorization' } });
}

// ---- Router ----
export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === 'OPTIONS') return json({}, 204);

    const url = new URL(req.url);
    const path = url.pathname;

    // ---- Google OAuth ----
    if (path === '/auth/google') {
      const redirect = `${env.ORIGIN}/auth/google/callback`;
      const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${env.GOOGLE_CLIENT_ID}&redirect_uri=${redirect}&response_type=code&scope=email%20profile`;
      return Response.redirect(authUrl, 302);
    }
    if (path === '/auth/google/callback') {
      const code = url.searchParams.get('code');
      if (!code) return json({ error: 'no code' }, 400);

      // Exchange code for tokens
      const tokenResp = await fetch('https://oauth2.googleapis.com/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ code, client_id: env.GOOGLE_CLIENT_ID, client_secret: env.GOOGLE_CLIENT_SECRET, redirect_uri: `${env.ORIGIN}/auth/google/callback`, grant_type: 'authorization_code' }),
      });
      const tokens: any = await tokenResp.json();
      if (!tokens.access_token) return json({ error: 'token exchange failed' }, 400);

      // Get user info
      const userResp = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', { headers: { Authorization: `Bearer ${tokens.access_token}` } });
      const userInfo: any = await userResp.json();
      if (!userInfo.email) return json({ error: 'no email' }, 400);

      // Upsert user in D1
      const { results } = await env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(userInfo.email).all();
      let userId: number;
      if (results.length === 0) {
        const r = await env.DB.prepare('INSERT INTO users (email, name, google_id) VALUES (?, ?, ?) RETURNING id').bind(userInfo.email, userInfo.name || '', userInfo.id).first();
        userId = r!.id as number;
      } else {
        userId = results[0].id as number;
        await env.DB.prepare('UPDATE users SET last_login = datetime(\'now\') WHERE id = ?').bind(userId).run();
      }

      const token = await createJWT({ userId, email: userInfo.email }, env.JWT_SECRET);
      return Response.redirect(`${env.ORIGIN}/?token=${token}`, 302);
    }

    // ---- Auth check ----
    const authHeader = req.headers.get('Authorization');
    let currentUser: Record<string, any> | null = null;
    if (authHeader?.startsWith('Bearer ')) {
      currentUser = await verifyJWT(authHeader.slice(7), env.JWT_SECRET);
    }

    // ---- Model storage ----
    if (path === '/models' && req.method === 'POST' && currentUser) {
      const body: any = await req.json();
      const r2Key = `models/${currentUser.userId}/${Date.now()}.json`;
      await env.MODELS.put(r2Key, JSON.stringify(body.adapter));
      const r = await env.DB.prepare('INSERT INTO models (user_id, name, r2_key, adapters, metadata) VALUES (?, ?, ?, ?, ?) RETURNING id')
        .bind(currentUser.userId, body.name || 'untitled', r2Key, JSON.stringify(body.adapter), JSON.stringify(body.metadata || {}))
        .first();
      return json({ id: r?.id, r2Key }, 201);
    }
    if (path === '/models' && req.method === 'GET' && currentUser) {
      const { results } = await env.DB.prepare('SELECT id, name, created_at, metadata FROM models WHERE user_id = ? ORDER BY created_at DESC').bind(currentUser.userId).all();
      return json(results);
    }
    if (path.startsWith('/models/') && req.method === 'GET' && currentUser) {
      const id = path.split('/')[2];
      const r: any = await env.DB.prepare('SELECT * FROM models WHERE id = ? AND user_id = ?').bind(parseInt(id), currentUser.userId).first();
      if (!r) return json({ error: 'not found' }, 404);
      const obj = await env.MODELS.get(r.r2_key);
      const adapter = obj ? await obj.text() : '{}';
      return json({ id: r.id, name: r.name, adapter: JSON.parse(adapter), metadata: JSON.parse(r.metadata || '{}'), created_at: r.created_at });
    }

    // ---- Data contribution ----
    if (path === '/data' && req.method === 'POST') {
      const body: any = await req.json();
      const userId = currentUser?.userId || null;
      await env.DB.prepare('INSERT INTO training_data (user_id, substrate, anchor_count, total_patches, median_de, p95_de, max_de) VALUES (?, ?, ?, ?, ?, ?, ?)')
        .bind(userId, body.substrate, body.anchorCount, body.totalPatches, body.medianDE, body.p95DE, body.maxDE).run();
      // Store detailed CGATS if provided (for retraining)
      if (body.uCgats && body.lCgats && userId) {
        const key = `cgats/${userId}/${Date.now()}`;
        await env.DATA.put(`${key}_u.txt`, body.uCgats);
        await env.DATA.put(`${key}_l.txt`, body.lCgats);
      }
      return json({ ok: true }, 201);
    }

    // ---- Health ----
    if (path === '/health') return json({ status: 'ok', env: env.ENV });

    return json({ error: 'not found' }, 404);
  },
} satisfies ExportedHandler<Env>;
