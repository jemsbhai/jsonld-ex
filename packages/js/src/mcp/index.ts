
import { JsonLdExMcpServer } from './server.js';

async function main() {
    const server = new JsonLdExMcpServer();
    await server.run();
}

main().catch(console.error);
