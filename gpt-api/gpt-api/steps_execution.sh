#Step 1: Create EKS Cluster
eksctl create cluster --name gpt-cluster-lat --region us-east-2 --nodegroup-name gpt-nodegroup-lat --nodes 3 --nodes-min 1 --nodes-max 4 --managed

#Step 2: Configure kubectl
aws eks --region us-east-2 update-kubeconfig --name gpt-cluster-lat

#Step 3: Create ECR Repository
aws ecr create-repository --repository-name gpt-api

#Step 4: Set up AWS CodeBuild
setup the code builder(which internally uses your buildspec.yml)
then run 


#Site 5: Deploy application
kubectl apply -f k8s/deployment.yaml   
kubectl apply -f k8s/service.yaml   


gpt-api/
│
├── main.py                  # FastAPI app code (defines the API routes and GPT logic)
├── Dockerfile               # Docker build config (for containerizing the app)
├── requirements.txt         # Python dependencies (e.g., fastapi, openai, uvicorn)
│-- buildspec.yml 
├── k8s/                     # Kubernetes config files
│   ├── deployment.yaml      # Describes the Deployment: replicas, container image, ports, etc.
│   └── service.yaml         # Exposes the Deployment via a Kubernetes Service

