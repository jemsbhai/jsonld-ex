
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import client from '../client.js';

export class JsonLdExMcpServer {
    private server: Server;

    constructor() {
        this.server = new Server(
            {
                name: "@jsonld-ex/core",
                version: "0.1.0",
            },
            {
                capabilities: {
                    tools: {},
                },
            }
        );

        this.setupHandlers();

        // Error handling
        this.server.onerror = (error) => console.error("[MCP Error]", error);
    }

    private setupHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: "jsonld_annotate",
                    description: "Annotate a value with AI/ML provenance metadata (confidence, source, method, etc.).",
                    inputSchema: {
                        type: "object",
                        properties: {
                            value: {
                                type: ["string", "number", "boolean"],
                                description: "The value to annotate"
                            },
                            metadata: {
                                type: "object",
                                properties: {
                                    confidence: { type: "number", minimum: 0, maximum: 1 },
                                    source: { type: "string" },
                                    method: { type: "string" },
                                    extractedAt: { type: "string", format: "date-time" },
                                    humanVerified: { type: "boolean" }
                                },
                                required: []
                            }
                        },
                        required: ["value", "metadata"]
                    },
                },
                {
                    name: "jsonld_merge",
                    description: "Merge multiple JSON-LD graphs with conflict resolution.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            graphs: {
                                type: "array",
                                items: { type: "object" },
                                description: "Array of JSON-LD documents/graphs to merge"
                            },
                            strategy: {
                                type: "string",
                                enum: ["highest", "weighted_vote", "recency", "union"],
                                description: "Conflict resolution strategy (default: highest)"
                            }
                        },
                        required: ["graphs"]
                    },
                },
                {
                    name: "jsonld_diff",
                    description: "Compute semantic difference between two JSON-LD graphs.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            graphA: { type: "object", description: "First graph" },
                            graphB: { type: "object", description: "Second graph" }
                        },
                        required: ["graphA", "graphB"]
                    },
                },
                {
                    name: "jsonld_propagate",
                    description: "Propagate confidence scores through a chain of processing steps.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            chain: {
                                type: "array",
                                items: { type: "number" },
                                description: "List of confidence scores in the chain"
                            },
                            method: {
                                type: "string",
                                enum: ["multiply", "min", "average"],
                                description: "Propagation method (default: multiply)"
                            }
                        },
                        required: ["chain"]
                    },
                },
                {
                    name: "jsonld_temporal",
                    description: "Add temporal qualifiers (@validFrom, @validUntil, @asOf) to a value.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            value: { description: "The value or node to qualify" },
                            options: {
                                type: "object",
                                properties: {
                                    validFrom: { type: "string", format: "date-time" },
                                    validUntil: { type: "string", format: "date-time" },
                                    asOf: { type: "string", format: "date-time" }
                                }
                            }
                        },
                        required: ["value", "options"]
                    },
                }
            ],
        }));

        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const { name, arguments: args } = request.params;

            try {
                switch (name) {
                    case "jsonld_annotate": {
                        const { value, metadata } = args as any;
                        const result = client.annotate(value, metadata);
                        return {
                            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                        };
                    }

                    case "jsonld_merge": {
                        const { graphs, strategy } = args as any;
                        const result = client.merge(graphs, { conflictStrategy: strategy });
                        return {
                            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                        };
                    }

                    case "jsonld_diff": {
                        const { graphA, graphB } = args as any;
                        const result = client.diff(graphA, graphB);
                        return {
                            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                        };
                    }

                    case "jsonld_propagate": {
                        const { chain, method } = args as any;
                        const result = client.propagate(chain, method);
                        return {
                            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                        };
                    }

                    case "jsonld_temporal": {
                        const { value, options } = args as any;
                        const result = client.addTemporal(value, options);
                        return {
                            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                        };
                    }

                    default:
                        throw new Error(`Unknown tool: ${name}`);
                }
            } catch (error: any) {
                return {
                    content: [{ type: "text", text: `Error: ${error.message}` }],
                    isError: true,
                };
            }
        });
    }

    async run() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error("JSON-LD-Ex MCP Server running on stdio");
    }
}
