/**
 * Type declarations for jsonld.js (Digital Bazaar)
 * Minimal declarations covering the APIs we use.
 */
declare module 'jsonld' {
  interface DocumentLoader {
    (url: string): Promise<{
      contextUrl: string | null;
      document: any;
      documentUrl: string;
    }>;
  }

  interface Options {
    base?: string;
    documentLoader?: DocumentLoader;
    expandContext?: any;
    [key: string]: any;
  }

  interface ToRdfOptions extends Options {
    format?: string;
  }

  export function expand(input: any, options?: Options): Promise<any[]>;
  export function compact(input: any, ctx: any, options?: Options): Promise<any>;
  export function flatten(input: any, ctx?: any, options?: Options): Promise<any>;
  export function toRDF(input: any, options?: ToRdfOptions): Promise<any>;
  export function fromRDF(input: any, options?: Options): Promise<any[]>;

  export const documentLoaders: {
    node(): DocumentLoader;
    xhr(): DocumentLoader;
  };
}
