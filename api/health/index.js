const { app } = require('@azure/functions');

app.http('health', {
    methods: ['GET'],
    authLevel: 'anonymous',
    handler: async (request, context) => {
        context.log('HTTP trigger function processed a health check request.');

        return {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                service: 'Cultivate Learning ML API',
                version: '1.0.0',
                environment: process.env.NODE_ENV || 'development'
            })
        };
    }
});