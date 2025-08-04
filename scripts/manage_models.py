"""
MLflow Model Registry Management Script

This script demonstrates comprehensive model lifecycle management using MLflow Model Registry:
1. List all registered models and their versions
2. Transition model stages (None → Staging → Production → Archived)
3. Show model lineage and metadata retrieval
4. Model comparison and deployment strategies

Following the coding guidelines:
- Explicit type annotations for all functions and variables
- Comprehensive documentation for model lifecycle operations
- Educational examples of production model management
"""

import argparse
import datetime
from typing import List, Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import RegisteredModel, ModelVersion


def setup_mlflow_client(tracking_uri: str = "http://localhost:5004") -> MlflowClient:
    """
    Configure and return MLflow client for registry operations.
    
    Args:
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Configured MLflow client instance
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    return client


def list_all_models(client: MlflowClient) -> None:
    """
    List all registered models and their versions with comprehensive metadata.
    
    Args:
        client: MLflow client instance
    """
    print("📋 REGISTERED MODELS OVERVIEW")
    print("=" * 80)
    
    try:
        registered_models: List[RegisteredModel] = client.search_registered_models()
        
        if not registered_models:
            print("No models found in the registry.")
            print("💡 Run the training script first: python scripts/train_iris_model.py")
            return
        
        for model in registered_models:
            print(f"\n🤖 Model: {model.name}")
            print(f"   Description: {model.description or 'No description'}")
            print(f"   Creation Time: {model.creation_timestamp}")
            print(f"   Last Updated: {model.last_updated_timestamp}")
            
            # Get all versions for this model
            model_versions: List[ModelVersion] = client.get_registered_model(model.name).latest_versions
            
            if model_versions:
                print(f"   📦 Versions ({len(model_versions)} total):")
                
                # Sort versions by version number
                sorted_versions = sorted(model_versions, key=lambda v: int(v.version))
                
                for version in sorted_versions:
                    stage_emoji = {
                        "None": "🔧",
                        "Staging": "🧪", 
                        "Production": "🚀",
                        "Archived": "📦"
                    }.get(version.current_stage, "❓")
                    
                    print(f"      {stage_emoji} Version {version.version} [{version.current_stage}]")
                    print(f"         Created: {version.creation_timestamp}")
                    print(f"         Status: {version.status}")
                    print(f"         Source: {version.source}")
                    
                    # Try to get aliases for this version (MLflow 3.x feature)
                    try:
                        # Get all aliases for the model
                        registered_model = client.get_registered_model(model.name)
                        version_aliases = []
                        
                        # Look for aliases pointing to this version
                        if hasattr(registered_model, 'aliases') and registered_model.aliases:
                            for alias_name, alias_version in registered_model.aliases.items():
                                if alias_version == version.version:
                                    version_aliases.append(alias_name)
                        
                        if version_aliases:
                            aliases_str = ", ".join([f"@{alias}" for alias in version_aliases])
                            print(f"         🏷️  Aliases: {aliases_str}")
                            
                    except Exception:
                        # Aliases not supported or other error - skip silently
                        pass
                    
                    # Get run info if available
                    if version.run_id:
                        try:
                            run = client.get_run(version.run_id)
                            metrics = run.data.metrics
                            params = run.data.params
                            
                            if metrics:
                                print(f"         📊 Metrics: ", end="")
                                metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
                                print(", ".join(metric_strs))
                            
                            if params:
                                print(f"         ⚙️  Params: ", end="")
                                param_strs = [f"{k}={v}" for k, v in params.items()]
                                print(", ".join(param_strs))
                                
                        except Exception as e:
                            print(f"         ⚠️  Could not retrieve run info: {e}")
            else:
                print("   📦 No versions found")
                
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        print("💡 Make sure MLflow server is running: mlflow server --host 0.0.0.0 --port 5004")


def set_model_alias(
    client: MlflowClient, 
    model_name: str, 
    version: str, 
    alias: str,
    remove_existing: bool = True
) -> None:
    """
    Set an alias for a model version (modern replacement for stage transitions).
    
    Args:
        client: MLflow client instance
        model_name: Name of the registered model
        version: Version to assign alias to
        alias: Target alias (e.g., "staging", "production", "champion")
        remove_existing: Whether to remove existing alias assignments
    """
    print(f"\n🔄 ALIAS ASSIGNMENT: {model_name} v{version} → @{alias}")
    print("=" * 60)
    
    try:
        # Check if model and version exist
        try:
            model_version = client.get_model_version(model_name, version)
            print(f"Model version {version} found with status: {model_version.status}")
                
        except Exception as e:
            print(f"❌ Model version not found: {model_name} v{version}")
            print(f"Error: {e}")
            return
        
        # Remove existing alias if requested and if it exists
        if remove_existing:
            try:
                # Check if alias already exists
                existing_version = client.get_model_version_by_alias(model_name, alias)
                if existing_version:
                    print(f"📝 Removing existing alias '{alias}' from version {existing_version.version}")
                    client.delete_registered_model_alias(model_name, alias)
            except Exception:
                # Alias doesn't exist, which is fine
                pass
        
        # Set the new alias
        print(f"🚀 Setting alias '{alias}' for version {version}...")
        client.set_registered_model_alias(model_name, alias, version)
        
        print(f"✅ Successfully set alias '{alias}' for {model_name} v{version}")
        
        # Verify the alias was set
        try:
            updated_version = client.get_model_version_by_alias(model_name, alias)
            print(f"✅ Verified: alias '{alias}' now points to version {updated_version.version}")
        except Exception as e:
            print(f"⚠️  Could not verify alias assignment: {e}")
        
    except Exception as e:
        print(f"❌ Error setting model alias: {e}")
        print("💡 Note: Model stages are deprecated in MLflow 3.x. Use aliases instead.")


def transition_model_stage(
    client: MlflowClient, 
    model_name: str, 
    version: str, 
    stage: str,
    archive_existing_prod: bool = True
) -> None:
    """
    DEPRECATED: Transition a model version to a new stage.
    
    This function is maintained for backward compatibility but maps to the new alias system.
    Stage transitions are deprecated in MLflow 3.x in favor of aliases.
    
    Args:
        client: MLflow client instance
        model_name: Name of the registered model
        version: Version to transition
        stage: Target stage ("None", "Staging", "Production", "Archived")
        archive_existing_prod: Whether to archive existing production models
    """
    print(f"\n⚠️  DEPRECATED: Stage transitions are deprecated in MLflow 3.x")
    print("💡 Using alias-based approach instead...")
    
    # Map stages to aliases
    stage_to_alias = {
        "Staging": "staging",
        "Production": "production", 
        "Archived": "archived"
    }
    
    if stage == "None":
        print(f"🔄 Removing any existing aliases for {model_name} v{version}")
        # Remove common aliases
        common_aliases = ["staging", "production", "champion", "latest"]
        for alias in common_aliases:
            try:
                existing_version = client.get_model_version_by_alias(model_name, alias)
                if existing_version and existing_version.version == version:
                    client.delete_registered_model_alias(model_name, alias)
                    print(f"✅ Removed alias '{alias}' from version {version}")
            except Exception:
                # Alias doesn't exist or points elsewhere
                pass
        return
    
    if stage not in stage_to_alias:
        print(f"❌ Invalid stage: {stage}")
        print(f"💡 Valid stages: None, Staging, Production, Archived")
        print(f"💡 Consider using set_model_alias() directly with custom aliases")
        return
    
    alias = stage_to_alias[stage]
    set_model_alias(client, model_name, version, alias, remove_existing=True)


def compare_model_performance(client: MlflowClient, model_name: str) -> None:
    """
    Compare performance metrics across all versions of a model.
    
    Args:
        client: MLflow client instance
        model_name: Name of the registered model
    """
    print(f"\n📊 PERFORMANCE COMPARISON: {model_name}")
    print("=" * 80)
    
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            print(f"No versions found for model: {model_name}")
            return
        
        print(f"{'Version':<8} {'Stage':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Created':<20}")
        print("-" * 80)
        
        # Sort by version number
        sorted_versions = sorted(model_versions, key=lambda v: int(v.version))
        
        for version in sorted_versions:
            try:
                run = client.get_run(version.run_id)
                metrics = run.data.metrics
                
                accuracy = metrics.get('accuracy', 0.0)
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1_score = metrics.get('f1_score', 0.0)
                
                created_date = datetime.datetime.fromtimestamp(
                    version.creation_timestamp / 1000
                ).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"{version.version:<8} {version.current_stage:<12} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f} {created_date:<20}")
                
            except Exception as e:
                print(f"{version.version:<8} {version.current_stage:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'Error':<20}")
        
        # Find best performing version
        best_version = None
        best_accuracy = 0.0
        
        for version in sorted_versions:
            try:
                run = client.get_run(version.run_id)
                accuracy = run.data.metrics.get('accuracy', 0.0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_version = version
            except:
                continue
        
        if best_version:
            print(f"\n🏆 Best performing version: {best_version.version} (Accuracy: {best_accuracy:.4f})")
            if best_version.current_stage != "Production":
                print(f"💡 Consider promoting v{best_version.version} to Production stage")
        
    except Exception as e:
        print(f"❌ Error comparing model performance: {e}")


def show_model_lineage(client: MlflowClient, model_name: str, version: str) -> None:
    """
    Show detailed lineage and metadata for a specific model version.
    
    Args:
        client: MLflow client instance
        model_name: Name of the registered model
        version: Model version to analyze
    """
    print(f"\n🔍 MODEL LINEAGE: {model_name} v{version}")
    print("=" * 80)
    
    try:
        model_version = client.get_model_version(model_name, version)
        
        print(f"📋 Basic Information:")
        print(f"   Model Name: {model_version.name}")
        print(f"   Version: {model_version.version}")
        print(f"   Current Stage: {model_version.current_stage}")
        print(f"   Status: {model_version.status}")
        print(f"   Creation Time: {datetime.datetime.fromtimestamp(model_version.creation_timestamp / 1000)}")
        print(f"   Source: {model_version.source}")
        print(f"   Run ID: {model_version.run_id}")
        
        if model_version.description:
            print(f"   Description: {model_version.description}")
        
        # Get detailed run information
        if model_version.run_id:
            print(f"\n🏃 Run Information:")
            run = client.get_run(model_version.run_id)
            
            print(f"   Experiment ID: {run.info.experiment_id}")
            print(f"   Run Name: {run.info.run_name or 'Unnamed'}")
            print(f"   Start Time: {datetime.datetime.fromtimestamp(run.info.start_time / 1000)}")
            print(f"   End Time: {datetime.datetime.fromtimestamp(run.info.end_time / 1000)}")
            print(f"   Status: {run.info.status}")
            
            # Parameters
            if run.data.params:
                print(f"\n⚙️  Training Parameters:")
                for key, value in run.data.params.items():
                    print(f"   {key}: {value}")
            
            # Metrics
            if run.data.metrics:
                print(f"\n📊 Performance Metrics:")
                for key, value in run.data.metrics.items():
                    print(f"   {key}: {value:.6f}")
            
            # Tags
            if run.data.tags:
                print(f"\n🏷️  Tags:")
                for key, value in run.data.tags.items():
                    if not key.startswith('mlflow.'):  # Skip system tags
                        print(f"   {key}: {value}")
        
        # Model artifacts
        print(f"\n📦 Model Artifacts:")
        try:
            artifacts = client.list_artifacts(model_version.run_id)
            for artifact in artifacts:
                print(f"   {artifact.path} ({artifact.file_size} bytes)")
        except Exception as e:
            print(f"   Could not list artifacts: {e}")
        
    except Exception as e:
        print(f"❌ Error showing model lineage: {e}")


def main() -> None:
    """Main function with command-line interface for model management operations."""
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python scripts/manage_models.py --list

  # Compare model performance
  python scripts/manage_models.py --compare iris-classifier-sklearn

  # Set alias for model version (recommended approach)
  python scripts/manage_models.py --alias iris-classifier-sklearn 1 staging
  python scripts/manage_models.py --alias iris-classifier-sklearn 1 production
  python scripts/manage_models.py --alias iris-classifier-sklearn 1 champion

  # Show model lineage
  python scripts/manage_models.py --lineage iris-classifier-sklearn 1

  # Transition model to staging (deprecated - use --alias instead)
  python scripts/manage_models.py --transition iris-classifier-sklearn 1 Staging
        """
    )
    
    parser.add_argument(
        '--tracking-uri', 
        default='http://localhost:5004',
        help='MLflow tracking server URI (default: http://localhost:5004)'
    )
    
    # Action arguments
    parser.add_argument('--list', action='store_true', help='List all registered models')
    parser.add_argument('--compare', metavar='MODEL_NAME', help='Compare performance across model versions')
    parser.add_argument('--lineage', nargs=2, metavar=('MODEL_NAME', 'VERSION'), help='Show model lineage and metadata')
    parser.add_argument('--alias', nargs=3, metavar=('MODEL_NAME', 'VERSION', 'ALIAS'), 
                       help='Set alias for model version (e.g., staging, production, champion)')
    parser.add_argument('--transition', nargs=3, metavar=('MODEL_NAME', 'VERSION', 'STAGE'), 
                       help='[DEPRECATED] Transition model to new stage - use --alias instead')
    parser.add_argument('--no-archive', action='store_true', 
                       help='Do not archive existing production models when promoting to production')
    
    args = parser.parse_args()
    
    # Setup MLflow client
    try:
        client = setup_mlflow_client(args.tracking_uri)
        print(f"🔗 Connected to MLflow at: {args.tracking_uri}")
    except Exception as e:
        print(f"❌ Failed to connect to MLflow: {e}")
        print("💡 Make sure MLflow server is running: mlflow server --host 0.0.0.0 --port 5004")
        return
    
    # Execute requested operation
    try:
        if args.list:
            list_all_models(client)
            
        elif args.compare:
            compare_model_performance(client, args.compare)
            
        elif args.lineage:
            model_name, version = args.lineage
            show_model_lineage(client, model_name, version)
            
        elif args.alias:
            model_name, version, alias = args.alias
            set_model_alias(client, model_name, version, alias)
            
        elif args.transition:
            model_name, version, stage = args.transition
            transition_model_stage(client, model_name, version, stage, not args.no_archive)
            
        else:
            print("No action specified. Use --help for usage information.")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()