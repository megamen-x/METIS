from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import desc

# Create a base class for declarative models
Base = declarative_base()

# Create a metadata instance for table definitions
metadata = MetaData()


class MeetingText(Base):
    """
    Model class representing the 'meetingText' table in the database.

    Attributes:
        id (int): Primary key for the table.
        user_id (int): Foreign key referencing the 'user' table, indexed for faster queries.
        decryption (str): Text field for storing the decrypted meeting text.
        official_protocol (str): Text field for storing the official protocol of the meeting.
        unofficial_protocol (str, optional): Text field for storing the unofficial protocol of the meeting.
    """
    __tablename__ = 'meetingText'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    decryption = Column(String, nullable=False)
    official_protocol = Column(String, nullable=False)
    unofficial_protocol = Column(String, nullable=True)


class CustomUser(Base):
    """
    Model class representing the 'user' table in the database.

    Attributes:
        user_id (int): Primary key for the table.
        username (str): Username of the user.
        email (str, optional): Email address of the user.
        app_key (str, optional): Application key for the user's email.
    """
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    username = Column(String, nullable=False)
    email = Column(String, nullable=True)
    app_key = Column(String, nullable=True)


class Database:
    """
    Class for managing the database connection and operations.
    """
    def __init__(self, db_path: str) -> None:
        """
        Initialize the database connection.

        Args:
            db_path (str): Path to the SQLite database file.
        """

        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_user(self, user_id: int, username: str) -> None:
        """
        Create a new user in the 'user' table if it doesn't already exist.

        Args:
            user_id (int): User ID of the new user.
            username (str): Username of the new user.
        """
        with self.Session() as session:
            new_user = session.query(CustomUser).filter_by(username=username).first()
            if new_user is None:
                new_user = CustomUser(user_id=user_id, username=username)    
                session.add(new_user)
                session.commit()
                

    def get_user(self, username: str) -> CustomUser:
        """
        Retrieve a user from the 'user' table by username.

        Args:
            username (str): Username of the user to retrieve.

        Returns:
            CustomUser: User object if found, 'Error' otherwise.
        """

        with self.Session() as session:
            record = (
                session.query(CustomUser)
                .filter(CustomUser.username == username)
                .first()
            )
            if record:
                return record
            return 'Error'

    def set_user_email(self, user_id: int, new_email: str, new_app_key: str) -> None:
        """
        Update the email and app_key fields for a user in the 'user' table.

        Args:
            user_id (int): User ID of the user to update.
            new_email (str): New email address for the user.
            new_app_key (str): New application key for the user's email.
        """

        with self.Session() as session:
            record = (
                session.query(CustomUser)
                .filter(CustomUser.user_id == user_id)
                .first()
            )
            record.email = new_email
            record.app_key = new_app_key
            session.commit()

    def create_meeting_text(self, user_id, decryption: str, off_doc: str, unoff_doc: str) -> None:
        """
        Create a new record in the 'meetingText' table.

        Args:
            user_id (int): User ID associated with the meeting text.
            decryption (str): Decrypted text of the meeting.
            off_doc (str): Official protocol of the meeting.
            unoff_doc (str): Unofficial protocol of the meeting.
        """

        with self.Session() as session:
            new_meeting = MeetingText(user_id=user_id, decryption=decryption, official_protocol=off_doc, unofficial_protocol=unoff_doc)
            session.add(new_meeting)
            session.commit()

    def get_meeting_text(self, user_id: int) -> list:
        """
        Retrieve the latest meeting text records for a user from the 'meetingText' table.

        Args:
            user_id (int): User ID to retrieve the meeting text for.

        Returns:
            list: A list containing the official protocol, unofficial protocol, and decryption text of the latest meeting.
        """
        
        with self.Session() as session:
            record = (
                session.query(MeetingText)
                .filter(MeetingText.user_id == user_id)
                .order_by(desc(MeetingText.id))
                .first()
            )
            if record:
                return [record.official_protocol, record.unofficial_protocol, record.decryption]
            return []