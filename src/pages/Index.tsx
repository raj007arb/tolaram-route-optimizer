import { useState, useEffect, useMemo } from "react";
import RouteMap from "@/components/RouteMap";
import RouteStats from "@/components/RouteStats";
import { locations, agents } from "@/data/locations";
import { solveVRPTSP, AgentRoute } from "@/utils/tspSolver";
import { MapPin, Route, Users, Zap } from "lucide-react";

const Index = () => {
  const [routes, setRoutes] = useState<AgentRoute[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<number | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(true);

  useEffect(() => {
    // Run optimization in a timeout to allow UI to render first
    const timer = setTimeout(() => {
      const optimizedRoutes = solveVRPTSP(locations, agents);
      setRoutes(optimizedRoutes);
      setIsOptimizing(false);
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  const stats = useMemo(() => {
    if (routes.length === 0) return null;
    const totalDistance = routes.reduce((sum, r) => sum + r.distance, 0);
    return {
      totalDistance,
      avgPerAgent: totalDistance / routes.length,
      totalLocations: locations.length,
      activeAgents: routes.length,
    };
  }, [routes]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                <Route className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold">TSP Route Optimizer</h1>
                <p className="text-xs text-muted-foreground">Multi-Agent Vehicle Routing</p>
              </div>
            </div>

            {stats && !isOptimizing && (
              <div className="hidden md:flex items-center gap-6">
                <div className="flex items-center gap-2 text-sm">
                  <MapPin className="w-4 h-4 text-primary" />
                  <span className="font-mono">{stats.totalLocations}</span>
                  <span className="text-muted-foreground">locations</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <Users className="w-4 h-4 text-accent" />
                  <span className="font-mono">{stats.activeAgents}</span>
                  <span className="text-muted-foreground">agents</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <Zap className="w-4 h-4 text-warning" />
                  <span className="font-mono">{stats.totalDistance.toFixed(1)}</span>
                  <span className="text-muted-foreground">km total</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-4 h-[calc(100vh-73px)]">
        {isOptimizing ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <h2 className="text-xl font-semibold mb-2">Optimizing Routes</h2>
              <p className="text-muted-foreground text-sm">
                Clustering {locations.length} locations across {agents.length} agents...
              </p>
            </div>
          </div>
        ) : (
          <div className="grid lg:grid-cols-[1fr,380px] gap-4 h-full">
            {/* Map */}
            <div className="glass-panel overflow-hidden h-[500px] lg:h-full animate-fade-in">
              <RouteMap
                routes={routes}
                selectedAgent={selectedAgent}
                onSelectAgent={setSelectedAgent}
              />
            </div>

            {/* Stats sidebar */}
            <div className="glass-panel overflow-hidden h-[500px] lg:h-full animate-slide-up">
              <RouteStats
                routes={routes}
                selectedAgent={selectedAgent}
                onSelectAgent={setSelectedAgent}
                totalLocations={locations.length}
              />
            </div>
          </div>
        )}
      </main>

      {/* Algorithm explanation */}
      <section className="container mx-auto px-4 py-8 border-t border-border/50">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-lg font-semibold mb-4">Optimization Approach</h2>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="glass-panel p-4">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center mb-3">
                <span className="font-bold text-primary">1</span>
              </div>
              <h3 className="font-medium mb-1">K-Means Clustering</h3>
              <p className="text-muted-foreground text-xs">
                Locations are partitioned into clusters using K-means++ initialization for balanced geographic distribution among agents.
              </p>
            </div>
            <div className="glass-panel p-4">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center mb-3">
                <span className="font-bold text-primary">2</span>
              </div>
              <h3 className="font-medium mb-1">Nearest Neighbor TSP</h3>
              <p className="text-muted-foreground text-xs">
                Each cluster is solved using nearest neighbor heuristic, building routes by always visiting the closest unvisited location.
              </p>
            </div>
            <div className="glass-panel p-4">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center mb-3">
                <span className="font-bold text-primary">3</span>
              </div>
              <h3 className="font-medium mb-1">2-Opt Improvement</h3>
              <p className="text-muted-foreground text-xs">
                Routes are refined using 2-opt local search, swapping edge pairs to eliminate route crossings and reduce total distance.
              </p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-4 text-center">
            All distances calculated using Haversine formula for accurate earth-curved paths • Routes form complete circuits returning to start
          </p>
        </div>
      </section>
    </div>
  );
};

export default Index;
