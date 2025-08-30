### Docker and Deployment

### Layered jar and Buildpacks image
```bash
./gradlew bootJar
./gradlew bootBuildImage --imageName=bookstore:0.0.1
docker run -p 8080:8080 bookstore:0.0.1
```

### Minimal Dockerfile (JVM)
```dockerfile
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY build/libs/*SNAPSHOT.jar app.jar
ENTRYPOINT ["java","-XX:InitialRAMPercentage=50.0","-XX:MaxRAMPercentage=80.0","-jar","/app/app.jar"]
```

### docker-compose for app + Postgres
```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: bookstore
    ports: ["5432:5432"]
  app:
    image: bookstore:0.0.1
    depends_on: [db]
    environment:
      SPRING_DATASOURCE_URL: jdbc:postgresql://db:5432/bookstore
      SPRING_DATASOURCE_USERNAME: postgres
      SPRING_DATASOURCE_PASSWORD: postgres
    ports: ["8080:8080"]
```

### Kubernetes (quick start)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: bookstore }
spec:
  replicas: 2
  selector: { matchLabels: { app: bookstore } }
  template:
    metadata: { labels: { app: bookstore } }
    spec:
      containers:
        - name: app
          image: bookstore:0.0.1
          ports: [{ containerPort: 8080 }]
          readinessProbe: { httpGet: { path: /actuator/health/readiness, port: 8080 } }
          livenessProbe: { httpGet: { path: /actuator/health/liveness, port: 8080 } }
---
apiVersion: v1
kind: Service
metadata: { name: bookstore }
spec:
  selector: { app: bookstore }
  ports: [{ port: 80, targetPort: 8080 }]
  type: ClusterIP
```

Next: `15-best-practices-and-checklists.md`.


