# HPC Time Priority Management Guide

## Understanding Time Priority

#### Definition

- Time priority: numerical score determining job scheduling order in HPC queue systems
- Higher priority jobs start before lower priority jobs
- Dynamic calculation updated continuously as queue conditions change

#### Core calculation formula

- Priority = (PriorityWeightAge × Age) + (PriorityWeightFairshare × FairshareScore) + (PriorityWeightJobSize × JobSizeScore) + (PriorityWeightPartition × PartitionScore) + (PriorityWeightQOS × QOSScore) + NiceScore

#### Key factors breakdown

- **Age factor**: linear increase over wait time
  - Prevents job starvation
  - Weight typically 1000-10000
- **Fairshare factor**: based on historical resource consumption
  - Users with less recent usage get higher scores
  - Calculated using decay formula over configured period
  - Encourages resource sharing among users
- **QOS weight**: quality of service multiplier
  - Premium QoS: 2.0-10.0x multiplier
  - Standard QoS: 1.0x baseline
  - Low priority QoS: 0.1-0.5x penalty
- **Partition weight**: queue-specific priority modifier
  - GPU partitions: often higher weight
  - Debug partitions: typically lower weight
  - Production partitions: balanced weight
- **Job size factor**: resource request impact
  - Large jobs: penalty to prevent monopolization
  - Small jobs: bonus for quick turnaround
  - Backfill consideration for efficient scheduling
- **Nice value**: user-adjustable priority
  - Range: -10000 to +10000
  - Negative values increase priority (requires privileges)
  - Positive values decrease priority

## Advanced Priority Optimization Strategies

#### Fairshare management

- **Monitor usage patterns**: track consumption via `sshare` command
  - Check personal fairshare score regularly
  - Monitor group/project fairshare status
  - Understand decay period (typically 14-30 days)
- **Strategic usage distribution**
  - Coordinate with team members on submission timing
  - Balance individual vs group fairshare impact
  - Use different accounts/projects if available
- **Historical usage analysis**
  - Review past month consumption patterns
  - Identify low-usage periods for strategic submissions
  - Plan large computational campaigns around fairshare recovery

#### QoS optimization techniques

- **QoS selection strategy**
  - Normal: standard priority, moderate limits
  - Long: extended walltime, lower priority
  - High: increased priority, tighter limits
  - Preempt: highest priority, can interrupt other jobs
- **QoS limit awareness**
  - Max concurrent jobs per QoS
  - Walltime restrictions per QoS level
  - Resource allocation caps (CPUs, memory, GPUs)
- **Multi-QoS job planning**
  - Submit urgent jobs to high-priority QoS
  - Use normal QoS for routine workloads
  - Reserve premium QoS for critical deadlines

#### Resource request optimization

- **Right-sizing strategy**
  - Profile applications to determine optimal resource needs
  - Avoid over-requesting CPUs/memory/GPUs
  - Use job arrays for embarrassingly parallel tasks
- **Walltime optimization**
  - Request minimum necessary runtime
  - Use checkpointing for long-running jobs
  - Split large jobs into smaller chunks when possible
- **Node sharing considerations**
  - Understand exclusive vs shared node policies
  - Optimize for backfill scheduling opportunities
  - Consider partial node usage for small jobs

#### Advanced scheduling techniques

- **Backfill exploitation**
  - Submit small, short jobs to fill gaps
  - Use `--begin` flag for delayed start times
  - Leverage reservation windows effectively
- **Dependency management**
  - Chain jobs using `--dependency` flags
  - Pipeline workflows for continuous processing
  - Use job arrays with dependencies for complex workflows
- **Preemption strategies**
  - Understand preemptible QoS options
  - Design checkpoint-restart capabilities
  - Use preemption-aware job submission patterns

## System-Specific Considerations

#### Stanage HPC specifics

- **GPU allocation priority**
  - A100 80GB: highest demand, plan accordingly
  - Submit GPU jobs during off-peak hours (weekends, holidays)
  - Consider CPU-only alternatives when possible
- **Partition characteristics**
  - GPU partitions: higher competition, longer queues
  - CPU partitions: more availability, faster turnaround
  - Interactive partitions: immediate access, limited resources
- **User limits and quotas**
  - Check disk quotas in home/scratch directories
  - Monitor CPU-hour allocations per project
  - Understand group resource sharing policies

#### Monitoring and diagnostics

- **Priority tracking commands**
  - `sprio`: show current job priorities
  - `squeue --format`: detailed queue information
  - `sshare`: fairshare scores and usage
  - `sacct`: historical job accounting data
- **Queue analysis tools**
  - `sinfo`: partition and node status
  - `scontrol show job`: detailed job information
  - `sstat`: running job resource usage
- **Automated monitoring setup**
  - Create scripts to track priority changes
  - Set up alerts for fairshare degradation
  - Monitor job completion rates and efficiency

## Troubleshooting Low Priority Issues

#### Common priority problems

- **Fairshare penalty recovery**
  - Stop submitting jobs temporarily to recover fairshare
  - Coordinate with group members to reduce collective usage
  - Wait for decay period to improve scores
- **QoS limit exceeded**
  - Check current job counts against QoS limits
  - Cancel unnecessary or failed jobs
  - Switch to different QoS with available slots
- **Resource contention**
  - Analyze queue composition for busy periods
  - Adjust resource requests to improve backfill chances
  - Consider alternative partitions or resources

#### Emergency priority boost options

- **Administrative requests**
  - Contact HPC support for urgent deadlines
  - Provide justification for priority increase
  - Request temporary QoS elevation
- **Resource reservation**
  - Reserve nodes for critical computation windows
  - Plan maintenance windows around reservations
  - Coordinate team usage of reserved resources

## Best Practices Summary

#### Daily operations

- Check fairshare status before large submissions
- Monitor queue status to identify optimal submission times
- Right-size resource requests based on actual needs
- Use appropriate QoS for job urgency level

#### Long-term strategy

- Maintain balanced resource usage patterns
- Develop checkpointing for long-running applications
- Build relationships with HPC support team
- Plan computational campaigns around system maintenance

#### Team coordination

- Share fairshare monitoring responsibilities
- Coordinate large job submissions
- Establish priority protocols for urgent work
- Cross-train on HPC optimization techniques

## References

- [Slurm Multifactor Priority Plugin Documentation](https://slurm.schedmd.com/multifactor_priorities.html)
- [Sheffield HPC Stanage User Guide](https://docs.hpc.shef.ac.uk/en/latest/stanage/)
- [Slurm Fair Tree Algorithm](https://slurm.schedmd.com/fair_tree.html)
- [Job Scheduling and Resource Management](https://slurm.schedmd.com/scheduling.html)
