# Build stage
# This stage installs dependencies and builds the application.
FROM node:18-alpine AS builder
WORKDIR /app
ENV NODE_ENV=production

# 1. Copy package files first to leverage Docker layer caching
COPY package.json package-lock.json* pnpm-lock.yaml* ./
RUN npm ci --silent

# 2. Copy source code last
# The 'public' folder, 'pages', 'components', etc., are copied here.
COPY . .

# 3. Build the Next.js app, which generates the .next folder
RUN npm run build


# Production image
# This stage is minimal and only includes files necessary for running the server.
FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
# expose port used by next start
EXPOSE 3000

# 4. Copy build artifacts from the 'builder' stage
# The 'builder' stage is defined above, and this is where COPY --from=builder belongs.
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
# Node modules are copied to run the built application
COPY --from=builder /app/node_modules ./node_modules 

# Default environment variable for backend communication
# This ensures the frontend client knows how to reach the orchestrator service inside Docker network
ENV NEXT_PUBLIC_API_URL="http://orchestrator:8000"

CMD ["npm", "run", "start"]