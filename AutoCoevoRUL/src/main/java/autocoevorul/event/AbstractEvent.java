package autocoevorul.event;

public abstract class AbstractEvent {

	private long creationTime;

	public AbstractEvent() {
		this.creationTime = System.currentTimeMillis();
	}

	public long getCreationTime() {
		return this.creationTime;
	}
}
