/**
 * Configuration management for Cultivate Learning ML MVP
 */

export interface APIConfig {
  host: string;
  port: number;
  debug: boolean;
}

export interface MLConfig {
  modelPath: string;
  batchSize: number;
  maxSequenceLength: number;
}

export interface AppConfig {
  api: APIConfig;
  ml: MLConfig;
  environment: 'development' | 'production' | 'test';
}

export const defaultConfig: AppConfig = {
  api: {
    host: process.env.API_HOST || 'localhost',
    port: parseInt(process.env.API_PORT || '8000', 10),
    debug: process.env.API_DEBUG === 'true',
  },
  ml: {
    modelPath: process.env.ML_MODEL_PATH || './models/',
    batchSize: parseInt(process.env.ML_BATCH_SIZE || '32', 10),
    maxSequenceLength: parseInt(process.env.ML_MAX_SEQUENCE_LENGTH || '512', 10),
  },
  environment: (process.env.NODE_ENV as 'development' | 'production' | 'test') || 'development',
};

export function getConfig(): AppConfig {
  return { ...defaultConfig };
}

export function validateConfig(config: AppConfig): boolean {
  return (
    config.api.port > 0 &&
    config.api.port < 65536 &&
    config.ml.batchSize > 0 &&
    config.ml.maxSequenceLength > 0
  );
}