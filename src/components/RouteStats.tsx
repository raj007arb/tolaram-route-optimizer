import { AgentRoute } from "@/utils/tspSolver";
import { ScrollArea } from "@/components/ui/scroll-area";

interface RouteStatsProps {
  routes: AgentRoute[];
  selectedAgent: number | null;
  onSelectAgent: (agentId: number | null) => void;
  totalLocations: number;
}

const RouteStats = ({ routes, selectedAgent, onSelectAgent, totalLocations }: RouteStatsProps) => {
  const totalDistance = routes.reduce((sum, r) => sum + r.distance, 0);
  const avgDistance = routes.length > 0 ? totalDistance / routes.length : 0;
  const maxDistance = Math.max(...routes.map((r) => r.distance), 0);
  const minDistance = Math.min(...routes.map((r) => r.distance), 0);
  const avgLocations = routes.length > 0 ? totalLocations / routes.length : 0;

  return (
    <div className="flex flex-col h-full">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-3 p-4">
        <div className="glass-panel p-4">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Total Distance</p>
          <p className="stat-value">{totalDistance.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">kilometers</p>
        </div>
        <div className="glass-panel p-4">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Avg / Agent</p>
          <p className="stat-value">{avgDistance.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">kilometers</p>
        </div>
        <div className="glass-panel p-4">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Locations</p>
          <p className="stat-value">{totalLocations}</p>
          <p className="text-xs text-muted-foreground">total stops</p>
        </div>
        <div className="glass-panel p-4">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Agents</p>
          <p className="stat-value">{routes.length}</p>
          <p className="text-xs text-muted-foreground">assigned</p>
        </div>
      </div>

      {/* Balance indicator */}
      <div className="px-4 pb-3">
        <div className="glass-panel p-3">
          <div className="flex justify-between items-center text-xs mb-2">
            <span className="text-muted-foreground">Distance Range</span>
            <span className="font-mono text-primary">
              {minDistance.toFixed(1)} - {maxDistance.toFixed(1)} km
            </span>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Avg: {avgLocations.toFixed(1)} stops/agent</span>
          </div>
        </div>
      </div>

      {/* Agent list */}
      <div className="flex-1 min-h-0 px-4 pb-4">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Agent Routes
        </h3>
        <ScrollArea className="h-full">
          <div className="space-y-2 pr-3">
            {routes.map((route) => (
              <button
                key={route.agentId}
                onClick={() => onSelectAgent(selectedAgent === route.agentId ? null : route.agentId)}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  selectedAgent === route.agentId
                    ? "bg-secondary border-primary/50"
                    : "bg-card/50 border-border/50 hover:bg-secondary/50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: route.color }}
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{route.agentName}</p>
                    <p className="text-xs text-muted-foreground">
                      {route.route.length} stops • {route.distance.toFixed(2)} km
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-mono text-primary">{route.distance.toFixed(1)}</p>
                    <p className="text-xs text-muted-foreground">km</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};

export default RouteStats;
