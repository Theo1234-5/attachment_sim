namespace AttachmentSim.Models;

public static class Sim
{
    public const int W = 660, H = 400;
    public const int CriticalPeriod = 300;
    public const int ChildPeriod    = 350;
    public const int AdultPeriod    = 550;
    public const int ElderPeriod    = 180;
    public const int SignalInterval = 35;
    public const int ResponseWindow = 24;
    public const double HebbianLR   = 0.018;
    public const double MaxW        = 2.2;
    public const double DecayRate   = 0.0018;
    public const int TaskInterval   = 60;
    public const double InheritNoise= 0.12;
    public const int DA=0, COR=1, OXY=2, SER=3;
}

public enum Stage { Infant, Child, Adult, Elder, Gone }
public enum AttachmentStyle { Unknown, Secure, Anxious, Avoidant, Disorganised }

public record StyleInfo(string Color, string Label, string Short);

public static class StyleMeta
{
    public static readonly Dictionary<AttachmentStyle, StyleInfo> Info = new()
    {
        [AttachmentStyle.Secure]       = new("#60efbc", "Secure",             "SEC"),
        [AttachmentStyle.Anxious]      = new("#f9a03f", "Anxious-Ambivalent", "ANX"),
        [AttachmentStyle.Avoidant]     = new("#7eb8f7", "Avoidant",           "AVO"),
        [AttachmentStyle.Disorganised] = new("#e07bda", "Disorganised",       "DIS"),
        [AttachmentStyle.Unknown]      = new("#555555", "Unclassified",       "???"),
    };
    public static StyleInfo Get(AttachmentStyle s) =>
        Info.TryGetValue(s, out var v) ? v : Info[AttachmentStyle.Unknown];
}

public enum RegionName { Amygdala, Hippocampus, Pfc, Acc }

public record RegionMeta(string Color, string Label, string Desc, double[] ChemSensitivity);

public static class RegionDefs
{
    public static readonly Dictionary<RegionName, RegionMeta> Meta = new()
    {
        [RegionName.Amygdala]    = new("#f87171", "Amygdala",      "Emotional feeling · love · fear",         new double[]{ +0.06, +0.20, +0.10, -0.08 }),
        [RegionName.Hippocampus] = new("#60a5fa", "Hippocampus",   "Memory · what worked · recall",           new double[]{ +0.12, -0.14, +0.08, +0.06 }),
        [RegionName.Pfc]         = new("#a78bfa", "Prefrontal",    "Regulation · impulse control · planning", new double[]{ +0.09, -0.18, +0.05, +0.12 }),
        [RegionName.Acc]         = new("#4ade80", "Ant. Cingulate","Empathy · mismatch · attunement",         new double[]{ +0.10, -0.10, +0.16, +0.07 }),
    };

    public static readonly (RegionName From, RegionName To, string Key, double Base)[] InterBase =
    {
        (RegionName.Amygdala,    RegionName.Hippocampus, "AMY->HIP",  0.20),
        (RegionName.Amygdala,    RegionName.Pfc,         "AMY->PFC",  0.15),
        (RegionName.Amygdala,    RegionName.Acc,         "AMY->ACC",  0.18),
        (RegionName.Hippocampus, RegionName.Pfc,         "HIP->PFC",  0.22),
        (RegionName.Hippocampus, RegionName.Acc,         "HIP->ACC",  0.16),
        (RegionName.Pfc,         RegionName.Amygdala,    "PFC->AMY", -0.25),
        (RegionName.Pfc,         RegionName.Acc,         "PFC->ACC",  0.20),
        (RegionName.Acc,         RegionName.Amygdala,    "ACC->AMY",  0.12),
        (RegionName.Acc,         RegionName.Hippocampus, "ACC->HIP",  0.14),
        (RegionName.Acc,         RegionName.Pfc,         "ACC->PFC",  0.18),
    };
}

public class RegionNet
{
    public const int N = 6;
    public double[,] W          = new double[N, N];
    public double[]  Activation = new double[N];
    public double[]  Threshold  = new double[N];
    public int[]     Refractory = new int[N];
    public double[]  Output     = new double[N];
    public double    Training   = 0;
    private static readonly Random _rng = new();

    public RegionNet(double[,]? parentW = null)
    {
        for (int i = 0; i < N; i++)
        {
            Threshold[i]  = 0.3 + _rng.NextDouble() * 0.15;
            Activation[i] = _rng.NextDouble() * 0.08;
        }
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            double fresh = -0.3 + _rng.NextDouble() * 0.8;
            if (parentW != null)
            {
                double parentSignal = parentW[i, j] * 0.20;
                double noise = (_rng.NextDouble() - 0.5) * Sim.InheritNoise * 3;
                fresh = Math.Clamp(fresh + parentSignal + noise, -Sim.MaxW, Sim.MaxW);
            }
            W[i, j] = fresh;
        }
    }

    public void Step(double[] ext, double chemGain)
    {
        var next = new double[N];
        for (int i = 0; i < N; i++)
        {
            if (Refractory[i] > 0) { Refractory[i]--; next[i] = Activation[i] * 0.7; continue; }
            double inp = Activation[i] * 0.82;
            for (int j = 0; j < N; j++)
                if (Activation[j] > Threshold[j] + chemGain)
                    inp += W[j, i] * Activation[j];
            if (ext.Length > i) inp += ext[i];
            inp = Math.Clamp(inp, 0, 1);
            next[i] = inp;
            if (inp >= Threshold[i] + chemGain) Refractory[i] = 3;
        }
        Array.Copy(next, Activation, N);
        Array.Copy(next, Output, N);
    }

    public void Hebbian(int[] pre, int[] post, double scale = 1)
    {
        foreach (int i in pre)
        foreach (int j in post)
        {
            if (i == j) continue;
            double d = Sim.HebbianLR * scale * Activation[i] * Activation[j];
            W[i, j] = Math.Clamp(W[i, j] + d, -Sim.MaxW, Sim.MaxW);
        }
        double tot = 0; int cnt = 0;
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) { tot += Math.Abs(W[i, j]); cnt++; }
        Training = Math.Min(1, (tot / cnt) / 0.8);
    }

    public double AvgActivation() => Activation.Average();
}

public class Brain
{
    public Dictionary<RegionName, RegionNet> Regions = new();
    public Dictionary<string, double>        Inter   = new();
    public double[] Chem = new double[4];
    private static readonly Random _rng = new();

    public Brain(Brain? parent = null)
    {
        foreach (RegionName rn in Enum.GetValues<RegionName>())
            Regions[rn] = new RegionNet(parent?.Regions[rn].W);

        foreach (var (_, _, key, baseW) in RegionDefs.InterBase)
        {
            double noise     = (_rng.NextDouble() - 0.5) * 0.12;
            double inherited = parent != null ? parent.Inter[key] * 0.15 + baseW * 0.85 : baseW;
            Inter[key] = Math.Clamp(inherited + noise, -Sim.MaxW, Sim.MaxW);
        }
        Chem = new double[] { 0.3, 0.2, 0.3, 0.5 };
    }

    public void Step(Dictionary<RegionName, double[]> signals)
    {
        var prev = new Dictionary<RegionName, double[]>();
        foreach (var rn in Regions.Keys)
            prev[rn] = (double[])Regions[rn].Output.Clone();

        foreach (RegionName rn in Enum.GetValues<RegionName>())
        {
            var meta = RegionDefs.Meta[rn];
            double g = 0;
            for (int ci = 0; ci < 4; ci++) g -= Chem[ci] * meta.ChemSensitivity[ci];
            double chemGain = Math.Clamp(g * 0.9, -0.25, 0.25);

            var ext = new double[RegionNet.N];
            if (signals.TryGetValue(rn, out var sig))
                for (int i = 0; i < Math.Min(sig.Length, RegionNet.N); i++) ext[i] += sig[i];

            foreach (var (from, to, key, _) in RegionDefs.InterBase)
            {
                if (to != rn) continue;
                double avg = prev[from].Average();
                double w   = Inter[key];
                for (int i = 0; i < 3; i++) ext[i] += w * avg * 0.4;
            }
            Regions[rn].Step(ext, chemGain);
        }

        double amyg = Regions[RegionName.Amygdala].AvgActivation();
        double pfc  = Regions[RegionName.Pfc].AvgActivation();
        double acc  = Regions[RegionName.Acc].AvgActivation();
        double hip  = Regions[RegionName.Hippocampus].AvgActivation();

        Chem[Sim.COR] = Math.Clamp(Chem[Sim.COR] + amyg*0.045 - pfc*0.032 - 0.007, 0, 1);
        Chem[Sim.OXY] = Math.Clamp(Chem[Sim.OXY] + acc*0.025  - 0.005,              0, 1);
        Chem[Sim.DA]  = Math.Clamp(Chem[Sim.DA]  + hip*0.020  - 0.005,              0, 1);
        Chem[Sim.SER] = Math.Clamp(Chem[Sim.SER] + pfc*0.015  - Chem[Sim.COR]*0.018,0, 1);

        foreach (var (from, to, key, baseW) in RegionDefs.InterBase)
        {
            double fa  = prev[from].Average();
            double ta  = Regions[to].AvgActivation();
            double sgn = Math.Sign(baseW);
            double d   = Sim.HebbianLR * 0.3 * fa * ta * sgn;
            Inter[key] = Math.Clamp(Inter[key] + d, -Sim.MaxW, Sim.MaxW);
        }
    }
}

public record IWM(double SelfWorth, double OtherWorth, double WorldSafety)
{
    public static IWM From(AttachmentStyle s) => s switch
    {
        AttachmentStyle.Secure       => new(0.82, 0.80, 0.76),
        AttachmentStyle.Anxious      => new(0.38, 0.52, 0.40),
        AttachmentStyle.Avoidant     => new(0.72, 0.22, 0.58),
        AttachmentStyle.Disorganised => new(0.28, 0.28, 0.20),
        _                            => new(0.50, 0.50, 0.50),
    };
}

public record ResponseEntry(bool Responded, int? Latency, int Tick);

public class DistressSignal { public string Type = "loneliness"; }

public record SimTask(
    string Id, string Label, string Prompt,
    string[] Options, int Correct, string Hint,
    RegionName Region);

public static class TaskDefs
{
    public static readonly SimTask[] All =
    {
        new("emo_read",  "Read the room",
            "Your child looks tense and withdrawn. What do they feel?",
            new[]{"Scared","Tired","Happy","Angry"}, 0,
            "Tension + withdrawal = fear", RegionName.Amygdala),
        new("emo_love",  "Express warmth",
            "Your child runs to you after being away. What do you feel?",
            new[]{"Joy & warmth","Annoyance","Indifference","Worry"}, 0,
            "Reunion activates attachment warmth", RegionName.Amygdala),
        new("emo_comf",  "Attune to distress",
            "Your child is crying but you don't know why. First response?",
            new[]{"Hold & soothe","Ask questions","Leave them","Distract"}, 0,
            "Emotional attunement comes before understanding", RegionName.Amygdala),
        new("mem_seq",   "Remember what worked",
            "Last time your child was scared at night, humming helped. They're scared again.",
            new[]{"Hum to them","Turn lights on","Leave them","Give food"}, 0,
            "Retrieve the successful strategy from memory", RegionName.Hippocampus),
        new("mem_ctx",   "Context recall",
            "Your child only cries before meals. They're crying now. Likely cause?",
            new[]{"Hunger","Fear","Loneliness","Pain"}, 0,
            "Context from past experience predicts current need", RegionName.Hippocampus),
        new("mem_pat",   "Pattern recognition",
            "Your child has been tired and irritable for 3 days. What pattern?",
            new[]{"Sleep regression","Illness onset","Normal mood","Boredom"}, 0,
            "Episodic memory detects recurring patterns", RegionName.Hippocampus),
        new("pfc_wait",  "Regulate yourself first",
            "You're stressed and exhausted. Your child cries. What do you do first?",
            new[]{"Breathe, then respond","Respond immediately","Ignore","Ask partner"}, 0,
            "PFC: regulate your own state before responding", RegionName.Pfc),
        new("pfc_inh",   "Override impulse",
            "Your child spills food again. Your impulse is frustration. You:",
            new[]{"Pause, then respond calmly","Express frustration","Leave","Raise voice"}, 0,
            "PFC inhibits amygdala's reactive impulse", RegionName.Pfc),
        new("pfc_plan",  "Plan ahead",
            "Your child struggles with transitions. Tomorrow is a big change. You:",
            new[]{"Prepare them with warnings","Wait and see","Distract","Ignore"}, 0,
            "Executive function: anticipate and plan", RegionName.Pfc),
        new("acc_mis",   "Detect mismatch",
            "Your child says 'I'm fine' but looks distressed. What do you trust?",
            new[]{"Body language","Their words","Neither","Ask someone"}, 0,
            "ACC detects mismatch between verbal and non-verbal signals", RegionName.Acc),
        new("acc_persp", "Take their perspective",
            "Your child is terrified of a dog. You know it's safe. You:",
            new[]{"Acknowledge fear, then reassure","Tell them not to be silly","Force them","Ignore"}, 0,
            "ACC: hold the child's perspective even when it differs from yours", RegionName.Acc),
        new("acc_tune",  "Fine-tune response",
            "You comforted your child but they're still upset. You:",
            new[]{"Try a different approach","Repeat what you did","Give up","Leave"}, 0,
            "ACC monitors outcome and flags when response needs adjustment", RegionName.Acc),
    };
}

public class Agent
{
    private static int _nextId = 0;
    private static readonly Random _rng = new();

    public int    Id  = _nextId++;
    public int    Gen;
    public double X, Y, Vx, Vy;
    public Brain  Brain;
    public Stage  Stage  = Stage.Infant;
    public int    Age    = 0;
    public AttachmentStyle AttachmentStyle = AttachmentStyle.Unknown;
    public IWM    Iwm    = IWM.From(AttachmentStyle.Unknown);
    public List<ResponseEntry> Record = new();
    public DistressSignal? Distress   = null;
    public int DistressTimer          = 0;
    public int NextSignalIn;
    public SimTask? CurrentTask       = null;
    public int TaskCooldown;
    public HashSet<string> TasksDone  = new();
    public double TaskScore           = 0;
    public Dictionary<int, double> Bonds = new();
    public List<(double X, double Y)> Trail = new();
    public double WanderAngle;
    public double Happiness = 0.5;
    public double Opacity   = 1;
    public bool   Recorded  = false;

    public static void ResetIds() => _nextId = 0;

    public Agent(int gen, double x, double y, Brain? parentBrain = null)
    {
        Gen          = gen;
        X = x; Y = y;
        Brain        = new Brain(parentBrain);
        NextSignalIn = 20 + _rng.Next(Sim.SignalInterval);
        TaskCooldown = Sim.TaskInterval;
        WanderAngle  = _rng.NextDouble() * Math.PI * 2;
    }

    public void Step(List<Agent> all, PlayerAction? action, int tick)
    {
        Age++;
        if (Stage == Stage.Gone) return;
        StageTransitions();
        if (Stage == Stage.Infant) StepInfant(action, tick);
        if (Stage == Stage.Child)  StepChild(action, all);
        if (Stage != Stage.Infant) StepBonds(all);
        StepBrain();
        StepMovement(all);
        UpdateHappiness();
        if (Stage == Stage.Elder) FadeOut();
    }

    private void StageTransitions()
    {
        if (Stage == Stage.Infant && Age >= Sim.CriticalPeriod)
        {
            AttachmentStyle = Classify();
            Iwm = IWM.From(AttachmentStyle);
            Brain.Chem[Sim.OXY] = Iwm.OtherWorth * 0.6;
            Brain.Chem[Sim.SER] = Iwm.SelfWorth  * 0.5;
            Stage = Stage.Child;
        }
        if (Stage == Stage.Child  && Age >= Sim.CriticalPeriod + Sim.ChildPeriod)  Stage = Stage.Adult;
        if (Stage == Stage.Adult  && Age >= Sim.CriticalPeriod + Sim.ChildPeriod + Sim.AdultPeriod) Stage = Stage.Elder;
        if (Stage == Stage.Elder  && Age >= Sim.CriticalPeriod + Sim.ChildPeriod + Sim.AdultPeriod + Sim.ElderPeriod) Stage = Stage.Gone;
    }

    private void StepInfant(PlayerAction? action, int tick)
    {
        NextSignalIn--;
        if (Distress != null)
        {
            DistressTimer++;
            bool responded = action?.TargetId == Id;
            if (responded)
            {
                Record.Add(new(true, DistressTimer, tick));
                Brain.Chem[Sim.OXY] = Math.Min(1, Brain.Chem[Sim.OXY] + 0.28);
                Brain.Chem[Sim.COR] = Math.Max(0, Brain.Chem[Sim.COR] - 0.22);
                Brain.Chem[Sim.DA]  = Math.Min(1, Brain.Chem[Sim.DA]  + 0.15);
                Brain.Regions[RegionName.Amygdala].Hebbian(new int[]{0,1}, new int[]{2,3,4,5}, 1.4);
                Brain.Regions[RegionName.Acc].Hebbian(new int[]{0,1}, new int[]{2,3,4,5}, 0.9);
                Distress = null; DistressTimer = 0;
                NextSignalIn = (int)(Sim.SignalInterval * 0.6 + _rng.NextDouble() * Sim.SignalInterval);
            }
            else if (DistressTimer > Sim.ResponseWindow)
            {
                Record.Add(new(false, null, tick));
                Brain.Chem[Sim.COR] = Math.Min(1, Brain.Chem[Sim.COR] + 0.18);
                Brain.Regions[RegionName.Amygdala].Hebbian(new int[]{0,1,2}, new int[]{0,1,2}, 0.9);
                Distress = null; DistressTimer = 0;
                NextSignalIn = (int)(Sim.SignalInterval * 0.4 + _rng.NextDouble() * Sim.SignalInterval);
            }
        }
        else if (NextSignalIn <= 0)
        {
            var types = new[]{"hunger","fear","loneliness","pain"};
            Distress = new DistressSignal { Type = types[_rng.Next(types.Length)] };
            DistressTimer = 0;
        }
    }

    private void StepChild(PlayerAction? action, List<Agent> all)
    {
        if (CurrentTask == null)
        {
            TaskCooldown--;
            if (TaskCooldown <= 0)
            {
                var weakest = Enum.GetValues<RegionName>()
                    .OrderBy(rn => Brain.Regions[rn].Training).First();
                var pool  = TaskDefs.All.Where(t => t.Region == weakest).ToArray();
                var avail = pool.Where(t => !TasksDone.Contains(t.Id) || pool.Length <= 1).ToArray();
                if (avail.Length > 0)
                {
                    CurrentTask  = avail[_rng.Next(avail.Length)];
                    TaskCooldown = Sim.TaskInterval;
                }
            }
        }
        if (CurrentTask != null)
        {
            if (Gen == 1 && action?.TaskAnswer is int ans)
                ApplyTaskAnswer(ans);
            else if (Gen > 1)
            {
                double skill = Brain.Regions[CurrentTask.Region].Training;
                bool correct = _rng.NextDouble() < (0.3 + skill * 0.55);
                ApplyTaskAnswer(correct ? CurrentTask.Correct : (CurrentTask.Correct + 1) % 4);
            }
        }
    }

    private void ApplyTaskAnswer(int ans)
    {
        if (CurrentTask == null) return;
        bool correct = ans == CurrentTask.Correct;
        double scale = correct ? 1.5 : 0.4;
        Brain.Regions[CurrentTask.Region].Hebbian(new int[]{0,1,2}, new int[]{3,4,5}, scale);
        if (correct) { TaskScore = Math.Min(1, TaskScore+0.07); Brain.Chem[Sim.DA]=Math.Min(1,Brain.Chem[Sim.DA]+0.1); }
        else Brain.Chem[Sim.COR] = Math.Min(1, Brain.Chem[Sim.COR]+0.04);
        TasksDone.Add(CurrentTask.Id);
        CurrentTask = null;
    }

    private void StepBonds(List<Agent> all)
    {
        foreach (var other in all)
        {
            if (other.Id == Id || other.Stage == Stage.Gone) continue;
            double dist = Math.Sqrt(Math.Pow(other.X-X,2)+Math.Pow(other.Y-Y,2));
            if (dist < 60)
            {
                Brain.Chem[Sim.OXY] = Math.Min(1, Brain.Chem[Sim.OXY]+0.025*Iwm.OtherWorth);
                Bonds[other.Id] = Math.Min(1,(Bonds.TryGetValue(other.Id,out var b)?b:0)+0.004);
                Brain.Regions[RegionName.Acc].Hebbian(new int[]{0,1}, new int[]{3,4,5}, 0.4);
            }
        }
        foreach (var key in Bonds.Keys.ToList())
            Bonds[key] = Math.Max(0, Bonds[key]-Sim.DecayRate);
    }

    private void StepBrain()
    {
        double edge = Math.Max(0,1-Math.Min(Math.Min(X,Sim.W-X),Math.Min(Y,Sim.H-Y))/70.0);
        var signals = new Dictionary<RegionName, double[]>
        {
            [RegionName.Amygdala]    = new double[]{ edge*0.4, Brain.Chem[Sim.COR]*0.2, 0,0,0,0 },
            [RegionName.Hippocampus] = new double[]{ Brain.Chem[Sim.DA]*0.15, 0,0,0,0,0 },
            [RegionName.Pfc]         = new double[]{ 0,0,0,0,0,0 },
            [RegionName.Acc]         = new double[]{ Brain.Chem[Sim.OXY]*0.2, 0,0,0,0,0 },
        };
        Brain.Step(signals);
    }

    private void StepMovement(List<Agent> all)
    {
        double amyg = Brain.Regions[RegionName.Amygdala].AvgActivation();
        double acc  = Brain.Regions[RegionName.Acc].AvgActivation();
        Agent? nearest = null; double nd = double.MaxValue;
        foreach (var o in all)
        {
            if (o.Id==Id||o.Stage==Stage.Gone) continue;
            double d=Math.Sqrt(Math.Pow(o.X-X,2)+Math.Pow(o.Y-Y,2));
            if(d<nd){nd=d;nearest=o;}
        }
        double dx=0,dy=0;
        if(nearest!=null&&nd>0)
        {
            double nx=(nearest.X-X)/nd, ny=(nearest.Y-Y)/nd;
            dx=nx*(acc*Iwm.OtherWorth - amyg*(1-Iwm.WorldSafety)*0.4);
            dy=ny*(acc*Iwm.OtherWorth - amyg*(1-Iwm.WorldSafety)*0.4);
        }
        WanderAngle+=(_rng.NextDouble()-0.5)*0.33;
        dx+=Math.Cos(WanderAngle)*0.25*Iwm.WorldSafety;
        dy+=Math.Sin(WanderAngle)*0.25*Iwm.WorldSafety;
        double spd=Stage==Stage.Infant?0.7:Stage==Stage.Elder?0.9:1.65;
        X=Math.Clamp(X+dx*spd,10,Sim.W-10);
        Y=Math.Clamp(Y+dy*spd,10,Sim.H-10);
        Trail.Add((X,Y));
        if(Trail.Count>18) Trail.RemoveAt(0);
    }

    private void UpdateHappiness() =>
        Happiness=Math.Clamp(Brain.Chem[Sim.SER]*0.35+Brain.Chem[Sim.OXY]*0.40-Brain.Chem[Sim.COR]*0.30+0.20,0,1);

    private void FadeOut()
    {
        int ea=Age-(Sim.CriticalPeriod+Sim.ChildPeriod+Sim.AdultPeriod);
        Opacity=Math.Max(0.08,1.0-(double)ea/Sim.ElderPeriod);
    }

    private AttachmentStyle Classify()
    {
        if(Record.Count==0) return AttachmentStyle.Unknown;
        var responded=Record.Where(r=>r.Responded).ToList();
        double sensitivity=(double)responded.Count/Record.Count;
        var latencies=responded.Where(r=>r.Latency.HasValue).Select(r=>r.Latency!.Value).ToList();
        double avg=latencies.Count>0?latencies.Average():Sim.ResponseWindow;
        double variance=latencies.Count>1?latencies.Select(l=>Math.Pow(l-avg,2)).Average():0;
        double consistency=1-Math.Min(1,variance/150.0);
        double anxiety=(1-consistency)*0.6+(1-sensitivity)*0.4;
        double avoidance=1-sensitivity;
        if(anxiety<0.28&&avoidance<0.28) return AttachmentStyle.Secure;
        if(anxiety>0.45&&avoidance<0.50) return AttachmentStyle.Anxious;
        if(avoidance>0.50)               return AttachmentStyle.Avoidant;
        return AttachmentStyle.Disorganised;
    }
}

public class PlayerAction
{
    public int? TargetId   = null;
    public int? TaskAnswer = null;
}

public record LineageEntry(
    int Gen, AttachmentStyle Style, IWM Iwm,
    int Responded, int Total, bool IsPlayer,
    Dictionary<RegionName, double> RegionTraining);