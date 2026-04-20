using AttachmentSim.Models;

namespace AttachmentSim.Services;

public class SimulationService
{
    private static readonly Random _rng = new();

    public List<Agent>        Agents     { get; private set; } = new();
    public List<LineageEntry> Lineage    { get; private set; } = new();
    public int                Tick       { get; private set; } = 0;
    public int                CurrentGen { get; private set; } = 1;
    public string             Phase      { get; private set; } = "";
    public bool               Running    { get; set; } = true;
    public int                SelectedId { get; set; } = 0;

    private PlayerAction? _pendingAction;
    private HashSet<int>  _spawnedGens = new();

    public event Action? StateChanged;

    public SimulationService() => Reset();

    public void Reset()
    {
        Agent.ResetIds();
        Agents.Clear();
        Lineage.Clear();
        _spawnedGens.Clear();
        Tick = 0; CurrentGen = 1;
        var g1 = new Agent(1, Sim.W/2.0, Sim.H/2.0);
        Agents.Add(g1);
        SelectedId = g1.Id;
        Phase = "GEN 1 INFANT · Click your child when they pulse red";
    }

    public void QueueDistressResponse(int agentId) =>
        _pendingAction = new PlayerAction { TargetId = agentId };

    public void QueueTaskAnswer(int answer)
    {
        _pendingAction ??= new PlayerAction();
        _pendingAction.TaskAnswer = answer;
    }

    public void SelectNearest(double mx, double my)
    {
        Agent? best=null; double bd=double.MaxValue;
        foreach(var a in Agents)
        {
            if(a.Stage==Stage.Gone) continue;
            double d=Math.Sqrt(Math.Pow(a.X-mx,2)+Math.Pow(a.Y-my,2));
            if(d<bd){bd=d;best=a;}
        }
        if(best==null) return;
        SelectedId=best.Id;
        if(best.Distress!=null&&best.Gen==1)
            _pendingAction=new PlayerAction{TargetId=best.Id};
    }

    public SimTask? GetActiveTask() =>
        Agents.FirstOrDefault(a=>a.Gen==1&&a.Stage==Stage.Child)?.CurrentTask;

    public Agent? SelectedAgent => Agents.FirstOrDefault(a=>a.Id==SelectedId)??Agents.FirstOrDefault();
    public Agent? CurrentInfant => Agents.FirstOrDefault(a=>a.Stage==Stage.Infant);

    public void Tick_()
    {
        if(!Running) return;
        Tick++;
        var action=_pendingAction;
        _pendingAction=null;

        Agents.RemoveAll(a=>a.Stage==Stage.Gone);
        foreach(var agent in Agents) agent.Step(Agents,action,Tick);

        foreach(var a in Agents)
        {
            if(a.Stage==Stage.Child&&!a.Recorded)
            {
                a.Recorded=true;
                var rt=new Dictionary<RegionName,double>();
                foreach(RegionName rn in Enum.GetValues<RegionName>()) rt[rn]=a.Brain.Regions[rn].Training;
                Lineage.Add(new(a.Gen,a.AttachmentStyle,a.Iwm,
                    a.Record.Count(r=>r.Responded),a.Record.Count,a.Gen==1,rt));
            }
        }

        var infant=Agents.FirstOrDefault(a=>a.Stage==Stage.Infant);
        var caregiver=Agents.FirstOrDefault(a=>a.Stage==Stage.Adult);

        if(caregiver!=null&&infant==null&&!_spawnedGens.Contains(caregiver.Gen+1))
        {
            _spawnedGens.Add(caregiver.Gen+1);
            int next=caregiver.Gen+1;
            var newInfant=new Agent(next,
                caregiver.X+(_rng.NextDouble()-0.5)*40,
                caregiver.Y+(_rng.NextDouble()-0.5)*40,
                caregiver.Brain);
            Agents.Add(newInfant);
            CurrentGen=next;
            Phase=$"GEN {next} INFANT · {StyleMeta.Get(caregiver.AttachmentStyle).Label} parent now caregiving";
            SelectedId=newInfant.Id;
        }

        var cgiver=Agents.FirstOrDefault(a=>a.Stage==Stage.Adult);
        var infnt=Agents.FirstOrDefault(a=>a.Stage==Stage.Infant);
        if(cgiver!=null&&infnt!=null&&infnt.Distress!=null&&infnt.Gen>1)
        {
            double accT=cgiver.Brain.Regions[RegionName.Acc].Training;
            double pfcT=cgiver.Brain.Regions[RegionName.Pfc].Training;
            double amygT=cgiver.Brain.Regions[RegionName.Amygdala].Training;
            double pRespond=cgiver.Iwm.OtherWorth*0.60+accT*0.20+amygT*0.10+pfcT*0.10;
            double penalty=(1-cgiver.Iwm.OtherWorth)*16;
            int latency=(int)(3+penalty*(1-pfcT*0.4)+_rng.NextDouble()*5);
            if(_rng.NextDouble()<pRespond&&infnt.DistressTimer>=latency)
            {
                _pendingAction=new PlayerAction{TargetId=infnt.Id};
                double dx=infnt.X-cgiver.X,dy=infnt.Y-cgiver.Y;
                double d=Math.Max(Math.Sqrt(dx*dx+dy*dy),1);
                cgiver.X=Math.Clamp(cgiver.X+dx/d*2.2,10,Sim.W-10);
                cgiver.Y=Math.Clamp(cgiver.Y+dy/d*2.2,10,Sim.H-10);
            }
        }

        StateChanged?.Invoke();
    }
}
