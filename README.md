
# GenAI Playground

Playground for GenAI code, learning and exercises

## n8n workflows

### Environment

To run n8n, you need to set the following environment variables:

- `DOMAIN_NAME`: your domain name, e.g. `example.io`.
- `SUBDOMAIN`: your subdomain, e.g. `n8n`.
- `SSL_EMAIL`: your email address used for SSL certificate generation, e.g. `n8n@example.io`.
- `GITHUB_PASSWORD`: your GitHub personal access token.
- `GENERIC_TIMEZONE`: optionally, setup your timezone, e.g. `Europe/Berlin`.

### Usage

1. Create a DNS: `${SUBDOMAIN}.${DOMAIN_NAME}`, e.g. locally `hosts` can be used to add `127.0.0.1 n8n.example.io`.
2. Run `cd n8n && make start` to start n8n, and visit `https://n8n.example.io/`.

### Considerations

To run certain workflows you would need to have:

- n8n license key (community edition).
- OpenAI API access token, e.g. `OPENAI_API_KEY`, for OpenAI Chat Model.
- GitHub personal access token, e.g. `GITHUB_PASSWORD`, for GitHub MCP.
